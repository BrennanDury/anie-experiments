import os
import numpy as np
import torch
import ray
from ray import tune
from ray.tune import Checkpoint
from data import load_navier_stokes_tensor
from architectures import TrimTransformer, PatchwiseMLP, PatchwiseMLP_norm, PatchwiseMLP_act, TimestepwiseMLP, \
    PositionalEncoding, PositionalUnencoding, RoPENd, RoPENdUnencoding, \
    LearnedPositionalEncoding, LearnedPositionalUnencoding, DecoderWrapper
from kind_wrappers import AcausalWrapper, OneStepWrapper, GenerateWrapper
from architecture_wrappers import PicardIterations, ArbitraryIterations, \
    make_weight_shared_modules, make_weight_unshared_modules
from training import Pipeline, training_epoch, evaluation_epoch
from functools import partial
from pathlib import Path
import time
import tempfile, os
from itertools import product
import argparse

def derive_dependent_hparams(cfg: dict) -> dict:
    d_model = cfg["d_model"]

    if cfg["positional_encoding"] == "coordinate":
        cfg["encoder_output_dim"] = d_model - 3
        cfg["encoder_channels"] = [d_model - 3, d_model - 3]
    else:
        cfg["encoder_output_dim"] = d_model
        cfg["encoder_channels"] = [d_model, d_model]

    cfg["dim_feedforward"] = d_model * 4

    if cfg["decoder_kind"] == "timestepwise":
        cfg["decoder_channels"] = [64, 256]
    else:
        cfg["decoder_channels"] = cfg["encoder_channels"]

    if cfg["train_kind"] == "acausal":
        cfg["train_kind"] = "acausal_narrow"
        cfg["val_kind"] = "acausal_narrow"
    else:
        cfg["val_kind"] = "generate"

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

    N, H, W, Q = 128, 64, 64, 1

    if cfg["mlp_kind"] == "norm":
        MLPClass = PatchwiseMLP_norm
    elif cfg["mlp_kind"] == "act":
        MLPClass = PatchwiseMLP_act
    else:
        MLPClass = PatchwiseMLP

    encoder = MLPClass(input_dim=Q,
                       output_dim=cfg["encoder_output_dim"],
                       hidden_dims=cfg["encoder_channels"],
                       K=cfg["patch_shape"],
                       S=cfg["patch_shape"])

    Hp = H // cfg["patch_shape"][0]
    Wp = W // cfg["patch_shape"][1]
    patch_size = cfg["patch_shape"][0] * cfg["patch_shape"][1]

    if cfg["decoder_kind"] == "timestepwise":
        decoder = TimestepwiseMLP(in_shape=torch.Size([Hp, Wp, cfg["encoder_output_dim"]]),
                                  hidden_dims=cfg["decoder_channels"],
                                  out_shape=torch.Size([H, W, Q]))
    else:
        decoder = DecoderWrapper(MLPClass(input_dim=cfg["encoder_output_dim"],
                                          output_dim=Q * patch_size,
                                          hidden_dims=list(reversed(cfg["encoder_channels"])),
                                          K=[1, 1],
                                          S=[1, 1]),
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


    make_module = partial(TrimTransformer,
                          d_model=cfg["d_model"],
                          nhead=cfg["n_head"],
                          dim_feedforward=cfg["dim_feedforward"],
                          dropout=cfg["dropout"],
                          n_layers=cfg["n_layers"],
                          scale=scale,
                          input_dim=transformer_input_dim)

    if cfg["share"]:
        modules = make_weight_shared_modules(make_module, n_modules=cfg["n_modules"])
    else:
        modules = make_weight_unshared_modules(make_module, n_modules=cfg["n_modules"])

    if cfg["picard"]:
        model = PicardIterations(modules, q=Q, r=cfg["r"])
    else:
        model = ArbitraryIterations(modules)

    mask = None
    if cfg["train_kind"] == "one-step":  # only needed for one-step
        n_patches = Hp * Wp
        idx = torch.arange(n_tokens, dtype=torch.int32)
        block_size = n_patches
        mask_after = block_size * ((idx // block_size) + 1) - 1
        mask = mask_after.to(device)

    acausal_model = AcausalWrapper(model)
    one_step_model = OneStepWrapper(model, mask=mask)
    generate_model = GenerateWrapper(model)

    acausal_pipeline = Pipeline(acausal_model, encoder, decoder, pos_enc, pos_unenc).to(device)
    one_step_pipeline = Pipeline(one_step_model, encoder, decoder, pos_enc, pos_unenc).to(device)
    generate_pipeline = Pipeline(generate_model, encoder, decoder, pos_enc, pos_unenc).to(device)

    pipelines = {"acausal_narrow": acausal_pipeline, "acausal_wide": acausal_pipeline, "one-step": one_step_pipeline, "generate": generate_pipeline}
    train_pipeline = pipelines[cfg["train_kind"]]
    val_pipeline = pipelines[cfg["val_kind"]]

    l1 = list(pos_enc.parameters()) if cfg["positional_encoding"] == "learned" else []
    l2 = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(l1 + l2, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

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
        train_pipeline.train()
        if epoch <= cfg["warmup_epochs"]:
            train_loss = training_epoch(train_loader, train_pipeline, cfg["train_kind"], loss_fn, optim,
                                         scheduler=scheduler, grad_clip_norm=cfg["grad_clip_norm"], device=device, compute_initial_conditions_loss=cfg["compute_initial_conditions_train_loss"])
        else:
            train_loss = training_epoch(train_loader, train_pipeline, cfg["train_kind"], loss_fn, optim,
                                         grad_clip_norm=cfg["grad_clip_norm"], device=device, compute_initial_conditions_loss=cfg["compute_initial_conditions_train_loss"])
        val_pipeline.eval()
        with torch.no_grad():
            val_loss = evaluation_epoch(val_loader, val_pipeline, cfg["val_kind"], loss_fn, device=device, compute_initial_conditions_loss=cfg["compute_initial_conditions_val_loss"])
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train-kind", type=str, default="one-step", help="Train kind")
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
        "seed":        [0],
        "share":       [False],
        "picard":      [False],
        "d_model":     [60],
        "train_kind":  [args.train_kind],
    }

    TUNED_GRID = {
        "lr":                 [args.lr],
        "decoder_kind":       ["timestepwise", "patchwise"],
        "positional_encoding": ["coordinate", "rope", "learned"],
    }

    CONSTANT_PARAMS = {
        "data": "ns_data.mat",
        "batch_size": 32,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "min_lr": 1e-9,
        "T_max": 101,
        "start_factor": 1e-9,
        "warmup_epochs": 10,
        "n_timesteps": n_timesteps,
        "project_input": False,
        "n_head": 4,
        "dropout": 0.1,
        "n_layers": 4,
        "n_modules": 3,
        "r": 0.5,
        "patch_shape": [4, 4],
        "mlp_kind": "act",
        "compute_initial_conditions_train_loss": False,
        "compute_initial_conditions_val_loss": False
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
                "epochs": 313,
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
        inner_results = tuner.fit()

        try:
            df = inner_results.get_dataframe()
            csv_path = results_dir / f"tune_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Saved tuning results to {csv_path}")
        except Exception as e:
            print(f"[WARN] Could not save tuning results: {e}")

if __name__ == "__main__":
    main()