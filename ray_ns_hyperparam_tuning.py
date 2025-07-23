import os
import numpy as np
import torch
import math
import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, Checkpoint
from data import load_navier_stokes_tensor
from architectures import TrimTransformer, PatchwiseMLP, PatchwiseMLP_norm, TimestepwiseMLP, \
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


############################################################
# Utility helpers                                          #
############################################################

def derive_dependent_hparams(cfg: dict) -> dict:
    """Augment the config with parameters that are deterministic functions
    of the user-controlled hyper-parameters."""
    d_model = cfg["d_model"]

    # Positional encoding & encoder output dims -------------------------------------------------
    if cfg["positional_encoding"] == "coordinate":
        cfg["encoder_output_dim"] = d_model - 3
        cfg["encoder_channels"] = [d_model - 3, d_model - 3]
    else:
        cfg["encoder_output_dim"] = d_model
        cfg["encoder_channels"] = [d_model, d_model]

    # Feed-forward dim --------------------------------------------------------------------------
    cfg["dim_feedforward"] = d_model * 2

    # Encoder/decoder channels ------------------------------------------------------------------
    if cfg["decoder_kind"] == "timestepwise":
        cfg["decoder_channels"] = [64, 256]
    else:
        cfg["decoder_channels"] = cfg["encoder_channels"]

    # Validation pipeline kind ------------------------------------------------------------------
    cfg["val_kind"] = "acausal" if cfg["train_kind"] == "acausal" else "generate"
    return cfg


############################################################
# Main training function (executed inside Ray worker)      #
############################################################

def train_ns_ray(config):
    """Ray Train entry-point function. Builds the model according to *config*,
    trains for *config["epochs"]* epochs and reports the final validation loss."""

    # Make sure we start from a clean GPU
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 1. Prepare deterministic, complete config
    # ---------------------------------------------------------------------
    cfg = derive_dependent_hparams(dict(config))  # work on a local copy

    # Torch / numpy seeds
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    # ---------------------------------------------------------------------
    # 2. Data (prefer Ray Dataset shard if available)
    # ---------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    train_loader = train_shard.iter_torch_batches(batch_size=cfg["batch_size"])
    val_loader = val_shard.iter_torch_batches(batch_size=cfg["batch_size"])

    N, T, H, W, Q = 5000, 10, 64, 64, 1

    # ---------------------------------------------------------------------
    # 3. Model components --------------------------------------------------
    # ---------------------------------------------------------------------
    if cfg["norm_mlp"]:
        MLPClass = PatchwiseMLP_norm
    else:
        MLPClass = PatchwiseMLP

    # encoder
    encoder = MLPClass(input_dim=Q,
                       output_dim=cfg["encoder_output_dim"],
                       hidden_dims=cfg["encoder_channels"],
                       K=cfg["patch_shape"],
                       S=cfg["patch_shape"]).to(device)

    Hp = H // cfg["patch_shape"][0]
    Wp = W // cfg["patch_shape"][1]
    patch_size = cfg["patch_shape"][0] * cfg["patch_shape"][1]

    # decoder
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
                                 Q=Q, patch_shape=cfg["patch_shape"]).to(device)

    # positional encodings -------------------------------------------------
    time_width = cfg["n_timesteps"] + 1 if cfg["train_kind"] == "acausal" else cfg["n_timesteps"]
    if cfg["positional_encoding"] == "coordinate":
        pos_enc = PositionalEncoding(time_width, Hp, Wp)
        pos_unenc = PositionalUnencoding(time_width, Hp, Wp)
    elif cfg["positional_encoding"] == "rope":
        pos_enc = RoPENd(torch.Size([time_width, Hp, Wp, cfg["encoder_output_dim"]]))
        pos_unenc = RoPENdUnencoding(torch.Size([time_width, Hp, Wp, cfg["encoder_output_dim"]]))
    else:  # learned
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

    # ---------------------------------------------------------------------
    # 4. Pipeline wrappers -------------------------------------------------
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

    pipelines = {"acausal": acausal_pipeline, "one-step": one_step_pipeline, "generate": generate_pipeline}
    train_pipeline = pipelines[cfg["train_kind"]]
    val_pipeline = pipelines[cfg["val_kind"]]

    # ---------------------------------------------------------------------
    # 5. Optimizer & LR scheduler -----------------------------------------
    l1 = list(pos_enc.parameters()) if cfg["positional_encoding"] == "learned" else []
    l2 = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(l1 + l2, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # warm-up + cosine schedule
    if cfg["warmup_epochs"] > 0:
        batches_per_epoch = math.ceil(N / cfg["batch_size"])
        warm_steps = batches_per_epoch * cfg["warmup_epochs"]
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=cfg["start_factor"], total_iters=warm_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["T_max"], eta_min=cfg["min_lr"])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_scheduler, cosine_scheduler], milestones=[warm_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["T_max"], eta_min=cfg["min_lr"])

    # ---------------------------------------------------------------------
    # 6. Training loop -----------------------------------------------------
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
                                         scheduler=scheduler, grad_clip_norm=cfg["grad_clip_norm"])
        else:
            train_loss = training_epoch(train_loader, train_pipeline, cfg["train_kind"], loss_fn, optim,
                                         grad_clip_norm=cfg["grad_clip_norm"])
        val_pipeline.eval()
        with torch.no_grad():
            val_loss = evaluation_epoch(val_loader, val_pipeline, cfg["val_kind"], loss_fn)
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
                train.report(metrics, checkpoint=ckpt)
        else:
            train.report(metrics)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


############################################################
# Search space & Ray Tune / Train orchestration            #
############################################################



def main():
    """Two-level hyper-parameter search.

    Outer loop  – enumerates the SEARCH parameters.
    Inner loop  – for each outer combination, grid-searches the TUNED parameters
                  for 200 epochs and picks the best combination.
    Long run    – repeats training for 10 000 epochs with the merged (outer +
                  best-tuned) configuration.
    """

    # ---------------------------------------------------------------------
    # 0. Shared Ray Dataset (loaded only once and placed in the object store)
    # ---------------------------------------------------------------------
    n_timesteps = 10
    init_conds, trajs = load_navier_stokes_tensor(Path("ns_data.mat"), n_timesteps=n_timesteps)

    items = [{"a": init_conds.numpy()[i], "u": trajs.numpy()[i]} for i in range(init_conds.shape[0])]
    ray_ds = ray.data.from_items(items)
    train, val = ray_ds.train_test_split(test_size=0.2)
    datasets = {"train": train, "val": val}
    # ---------------------------------------------------------------------
    # 1. Hyper-parameter grids
    # ---------------------------------------------------------------------
    SEARCH_GRID = {
        "seed":        [0, 1],
        "share":       [True, False],
        "picard":      [True, False],
        "d_model":     [60, 120, 240],
        "train_kind":  ["acausal", "one-step", "generate"],
    }

    TUNED_GRID = {
        "lr":                 [1e-3, 1e-4, 1e-5],
        "decoder_kind":       ["timestepwise", "patchwise"],
        "norm_mlp":           [True, False],
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
    }

    outer_keys = list(SEARCH_GRID.keys())

    # ---------------------------------------------------------------------
    # 2. Outer loop – iterate over SEARCH parameter combinations
    # ---------------------------------------------------------------------
    for outer_values in product(*(SEARCH_GRID[k] for k in outer_keys)):
        outer_cfg = dict(zip(outer_keys, outer_values))

        # -------------------- inner grid search (200 epochs) -------------
        tuned_space = {
            "train_loop_config": {
                # grid over the tuned parameters
                **{k: tune.grid_search(v) for k, v in TUNED_GRID.items()},
                # fixed SEARCH parameters for this outer iteration
                **outer_cfg,
                # constants
                **CONSTANT_PARAMS,
                "epochs": 5,
                "name": "inner_tune",
            }
        }

        trainer = TorchTrainer(
            train_loop_per_worker=train_ns_ray,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
            train_loop_config={},
            datasets=datasets,
        )

        tuner = tune.Tuner(
            trainer,
            param_space=tuned_space,
            tune_config=tune.TuneConfig(metric="val_loss", mode="min"),
        )

        inner_results = tuner.fit()
        best_inner_cfg_full = inner_results.get_best_result().config["train_loop_config"]

        best_tuned_subset = {k: best_inner_cfg_full[k] for k in TUNED_GRID.keys()}

        # -------------------- long run (10 000 epochs) --------------------
        long_cfg = {
            **outer_cfg,
            **best_tuned_subset,
            **CONSTANT_PARAMS,
            "epochs": 1,
            "name": "long_run",
        }

        long_trainer = TorchTrainer(
            train_loop_per_worker=train_ns_ray,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
            train_loop_config=long_cfg,
            datasets=datasets,
        )

        long_result = long_trainer.fit()
        final_val_loss = long_result.metrics.get("val_loss", None)
        print(f"Finished SEARCH combination {outer_cfg} → final val_loss={final_val_loss}")


if __name__ == "__main__":
    main()