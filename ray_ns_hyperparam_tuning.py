import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ray
from ray import tune
from ray.tune import Checkpoint
from data import load_navier_stokes_tensor
from architectures import PatchwiseMLP, PatchwiseMLP_norm, PatchwiseMLP_act, PatchwiseMLP_remove, TimestepwiseMLP, \
    PositionalEncoding, PositionalUnencoding, RoPENd, RoPENdUnencoding, \
    LearnedPositionalEncoding, LearnedPositionalUnencoding, DecoderWrapper, TransformerPipeline
from training import training_epoch, evaluation_epoch
from pathlib import Path
import time
import tempfile, os
from itertools import product
import argparse

def derive_dependent_hparams(cfg: dict) -> dict:    
    if cfg["positional_encoding"] == "coordinate":
        s = cfg["d_model"] - 3
    else:
        s = cfg["d_model"]
    t = s * cfg["encoder_ff_factor"]
    w = cfg["d_model"] * cfg["ff_factor"]
    cfg["encoder_output_dim"] = s
    cfg["encoder_channels"] = [t, t]
    cfg["dim_feedforward"] = w

    if cfg["decoder_kind"] == "timestepwise":
        cfg["decoder_channels"] = [64, 256]
    else:
        cfg["decoder_channels"] = cfg["encoder_channels"]

    if cfg["train_kind"] == "acausal":
        if cfg["narrow"]:
            cfg["train_kind"] = "acausal_narrow"
            cfg["val_kind"] = "acausal_narrow"
        else:
            cfg["train_kind"] = "acausal_wide"
            cfg["val_kind"] = "acausal_wide"
    else:
        cfg["val_kind"] = "generate"

    if cfg["model_activation"] == "ELU":
        cfg["model_activation"] = F.elu
    else:
        cfg["model_activation"] = F.relu

    if cfg["encoder_activation"] == "ELU":
        cfg["encoder_activation"] = nn.ELU
    else:
        cfg["encoder_activation"] = nn.ReLU

    if cfg["decoder_activation"] == "ELU":
        cfg["decoder_activation"] = nn.ELU
    else:
        cfg["decoder_activation"] = nn.ReLU
    return cfg


def train_fn(config, ds):
    torch.cuda.empty_cache()

    cfg = derive_dependent_hparams(dict(config))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_shard = ds["train"]
    val_shard = ds["val"]
    train_loader = train_shard.iter_torch_batches(batch_size=cfg["batch_size"], drop_last=True)
    val_loader = val_shard.iter_torch_batches(batch_size=cfg["batch_size"], drop_last=True)

    N, H, W, Q = 4000, 64, 64, 1

    if cfg["mlp_kind"] == "norm":
        MLPClass = PatchwiseMLP_norm
    elif cfg["mlp_kind"] == "act":
        MLPClass = PatchwiseMLP_act
    elif cfg["mlp_kind"] == "remove":
        MLPClass = PatchwiseMLP_remove
    else:
        MLPClass = PatchwiseMLP

    encoder = MLPClass(input_dim=Q,
                       output_dim=cfg["encoder_output_dim"],
                       hidden_dims=cfg["encoder_channels"],
                       K=cfg["patch_shape"],
                       S=cfg["patch_shape"],
                       activation=cfg["encoder_activation"])

    Hp = H // cfg["patch_shape"][0]
    Wp = W // cfg["patch_shape"][1]
    patch_size = cfg["patch_shape"][0] * cfg["patch_shape"][1]

    if cfg["decoder_kind"] == "timestepwise":
        decoder = TimestepwiseMLP(in_shape=torch.Size([Hp, Wp, cfg["encoder_output_dim"]]),
                                  hidden_dims=cfg["decoder_channels"],
                                  out_shape=torch.Size([H, W, Q]),
                                  activation=cfg["decoder_activation"])
    else:
        decoder = DecoderWrapper(MLPClass(input_dim=cfg["encoder_output_dim"],
                                          output_dim=Q * patch_size,
                                          hidden_dims=list(reversed(cfg["encoder_channels"])),
                                          K=[1, 1],
                                          S=[1, 1],
                                          activation=cfg["decoder_activation"]),
                                 Q=Q, patch_shape=cfg["patch_shape"])

    time_width = cfg["n_timesteps"] + 1 if cfg["train_kind"] == "acausal_wide" else cfg["n_timesteps"]
    if cfg["positional_encoding"] == "coordinate":
        pos_enc = PositionalEncoding(time_width, Hp, Wp)
        pos_unenc = PositionalUnencoding(time_width, Hp, Wp)
    elif cfg["positional_encoding"] == "rope":
        pos_enc = RoPENd(torch.Size([time_width, Hp, Wp, cfg["encoder_output_dim"]]))
        pos_unenc = RoPENdUnencoding(torch.Size([time_width, Hp, Wp, cfg["encoder_output_dim"]]))
    else:
        pos_enc = LearnedPositionalEncoding(torch.Size([time_width, Hp, Wp, cfg["encoder_output_dim"]]))
        pos_unenc = LearnedPositionalUnencoding(pos_enc)

    if cfg["project_input"]:
        transformer_input_dim = cfg["encoder_output_dim"] + (3 if cfg["positional_encoding"] == "coordinate" else 0)
    else:
        transformer_input_dim = None
        assert cfg["d_model"] == cfg["encoder_output_dim"] + (3 if cfg["positional_encoding"] == "coordinate" else 0)

    n_tokens = time_width * Hp * Wp
    scale = 1.0 / n_tokens

    model = TransformerPipeline(pos_enc,
                                pos_unenc,
                                d_model=cfg["d_model"],
                                nhead=cfg["n_head"],
                                dim_feedforward=cfg["dim_feedforward"],
                                dropout=cfg["dropout"],
                                n_layers=cfg["n_layers"],
                                scale=scale,
                                input_dim=transformer_input_dim,
                                activation=cfg["model_activation"],
                                n_modules=cfg["n_modules"],
                                inner_wrap=cfg["inner_wrap"],
                                outer_wrap=cfg["outer_wrap"],
                                share=cfg["share"],
                                encoder=encoder,
                                decoder=decoder)

    model = model.to(device)

    optim = torch.optim.Adam(list(model.parameters()), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["warmup_epochs"] > 0:
        batches_per_epoch = N // cfg["batch_size"]
        warm_steps = batches_per_epoch * cfg["warmup_epochs"]
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=cfg["start_factor"], total_iters=warm_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["T_max"], eta_min=cfg["min_lr"])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_scheduler, cosine_scheduler], milestones=[warm_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["T_max"], eta_min=cfg["min_lr"])

    train_losses = []
    val_losses = []
    epoch_times = []
    loss_fn = torch.nn.functional.mse_loss

    best_val_loss = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start_time = time.time()
        model.train()
        if epoch <= cfg["warmup_epochs"]:
            train_loss = training_epoch(train_loader, model,
                                        cfg["train_kind"],
                                        loss_fn,
                                        optim,
                                        scheduler=scheduler,
                                        grad_clip_norm=cfg["grad_clip_norm"],
                                        device=device,
                                        compute_initial_conditions_loss=cfg["compute_initial_conditions_train_loss"])
        else:
            train_loss = training_epoch(train_loader, model,
                                        cfg["train_kind"],
                                        loss_fn,
                                        optim,
                                        grad_clip_norm=cfg["grad_clip_norm"],
                                        device=device,
                                        compute_initial_conditions_loss=cfg["compute_initial_conditions_train_loss"])
        model.eval()

        with torch.no_grad():
            val_loss = evaluation_epoch(val_loader, model,
                                        cfg["val_kind"],
                                        loss_fn,
                                        device=device,
                                        compute_initial_conditions_loss=cfg["compute_initial_conditions_val_loss"])
        if epoch > cfg["warmup_epochs"]:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(time.time() - epoch_start_time)

        metrics = {
            "val_loss": val_loss,
            "train_loss": train_loss,
            "epoch": epoch,
            "epoch_time": epoch_times[-1]
        }

        if val_loss < best_val_loss:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "config": cfg,
                    },
                    os.path.join(tempdir, "state.pt"),
                )

                ckpt = Checkpoint.from_directory(tempdir)
                tune.report(metrics, checkpoint=ckpt)
        else:
            tune.report(metrics)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Navier–Stokes Ray Tune search")
    parser.add_argument("--results-dir", type=str, default="tune_results",
                        help="Directory where tuning result CSV files will be written.")
    args, _ = parser.parse_known_args()

    base_results_dir = Path(args.results_dir).resolve()
    base_results_dir.mkdir(parents=True, exist_ok=True)

    ray.init(object_store_memory=200_000_000_000)

    n_timesteps = 9
    init_conds, trajs = load_navier_stokes_tensor(Path("ns_data.mat"), n_timesteps=n_timesteps)

    items = [{"a": init_conds.numpy()[i], "u": trajs.numpy()[i]} for i in range(init_conds.shape[0])]
    ray_ds = ray.data.from_items(items)
    train, val = ray_ds.train_test_split(test_size=0.2)
    datasets = {"train": train, "val": val}

    SEARCH_GRID = {
        "train_kind": ["generate", "acausal", "one_step"],
        "share":      [True, False],
        "inner_wrap": [True, False],
    }

    TUNED_GRID = {
        "mlp_kind": ["norm", "act"],
        "positional_encoding": ["rope", "coordinate"],
        "decoder_kind": ["patchwise", "timestepwise"],
    }

    CONSTANT_PARAMS = {
        "data": "ns_data.mat",
        "model_activation": "ReLU",
        "encoder_activation": "ReLU",
        "decoder_activation": "ReLU",
        "n_timesteps": n_timesteps,
        "seed": 0,
        "d_model": 60,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "min_lr": 1e-9,
        "T_max": 101,
        "start_factor": 1e-9,
        "warmup_epochs": 10,
        "n_head": 4,
        "dropout": 0.1,
        "ff_factor": 4,
        "encoder_ff_factor": 4,
        "lr": 1e-3,
        "epochs": 313,
        "n_modules": 3,
        "n_layers": 4,
        "patch_shape": [4, 4],
        "project_input": False,
        "compute_initial_conditions_train_loss": False,
        "compute_initial_conditions_val_loss": False,
        "narrow": True,
        "outer_wrap": False,
    }

    outer_keys = list(SEARCH_GRID.keys())

    for outer_values in product(*(SEARCH_GRID[k] for k in outer_keys)):
        outer_cfg = dict(zip(outer_keys, outer_values))

        outer_tag = "__".join(f"{k}-{v}" for k, v in outer_cfg.items())
        results_dir = base_results_dir / outer_tag
        results_dir.mkdir(parents=True, exist_ok=True)

        param_space = {
                **{k: tune.grid_search(v) for k, v in TUNED_GRID.items()},
                **outer_cfg,
                **CONSTANT_PARAMS,
                "name": "inner_tune",
        }

        train_driver = tune.with_resources(train_fn, resources={"gpu": 1, "cpu": 11 })
        train_driver = tune.with_parameters(train_driver, ds=datasets)
        tune_config = tune.TuneConfig(metric="val_loss", mode="min")
        run_config = tune.RunConfig(storage_path=results_dir)
        tuner = tune.Tuner(
            train_driver,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
        tuner.fit()

if __name__ == "__main__":
    main()