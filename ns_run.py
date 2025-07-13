# %%
import torch
from torch.nn import functional as F
import numpy as np
import argparse
from pathlib import Path
from data import load_navier_stokes_tensor, setup_dataloaders
import time
from training import training_epoch, evaluation_epoch

parser = argparse.ArgumentParser(description="Navier–Stokes training script.")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--data", type=Path, default=Path("ns_data.mat"), help="Path to the .mat dataset.")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 penalty) for Adam optimizer.")
parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (set to 0 to disable).")
parser.add_argument("--min-lr", type=float, default=1e-9, help="Minimum learning rate for cosine annealing scheduler.")
parser.add_argument("--T-max", type=int, default=101, help="Maximum number of iterations for cosine annealing scheduler.")
parser.add_argument("--n-timesteps", type=int, default=11, help="Number of temporal frames to sample from the raw data (consistent with notebook).")

parser.add_argument("--share", action="store_true", help="Share weights between modules.")
parser.add_argument("--no-share", dest="share", action="store_false", help="Don't share weights between modules.")
parser.set_defaults(share=True)

parser.add_argument("--picard", action="store_true", help="Use Picard iterations.")
parser.add_argument("--no-picard", dest="picard", action="store_false", help="Don't use Picard iterations.")
parser.set_defaults(picard=True)

parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--dim_feedforward", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--n_modules", type=int, default=4)
parser.add_argument("--r", type=float, default=0.5)

# Encoder arguments
parser.add_argument("--encoder-hidden-dim", type=int, default=None, 
                    help="Hidden dimension for encoder (default: d_model - P)")
parser.add_argument("--encoder-hidden-ff", type=int, default=128,
                    help="Hidden feedforward dimension for encoder")
parser.add_argument("--patch_shape", type=int, nargs=2, default=[4, 4],
                    help="A token is a patch of size patch_shape")

# Decoder arguments
parser.add_argument("--decoder-hidden-channels", type=int, nargs="+", default=[64, 256],
                    help="Hidden channels for decoder MLP (excluding final output channel)")

parser.add_argument("--train-kind", choices=["acausal", "one_step", "generate"], default="acausal",
                    help="Pipeline kind to use during training")
parser.add_argument("--val-kind", choices=["acausal", "one_step", "generate"], default="acausal",
                    help="Pipeline kind to use during validation")

args = parser.parse_args()

# %%
# Create directory structure
Path("runs").mkdir(exist_ok=True)
base_dir = Path("runs/" + args.name)
base_dir.mkdir(exist_ok=True)

# Find the next available run number
run_num = 0
while True:
    run_dir = base_dir / f"run{run_num}"
    if not run_dir.exists():
        break
    run_num += 1

# Create the run directory
run_dir.mkdir(exist_ok=True)
print(f"Created run directory: {run_dir}")

# Save hyperparameters/config
config_dict = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'grad_clip_norm': args.grad_clip_norm,
    'min_lr': args.min_lr,
    'T_max': args.T_max,
    'n_timesteps': args.n_timesteps,
    'share': args.share,
    'picard': args.picard,
    'd_model': args.d_model,
    'nhead': args.nhead,
    'dim_feedforward': args.dim_feedforward,
    'dropout': args.dropout,
    'n_layers': args.n_layers,
    'n_modules': args.n_modules,
    'r': args.r,
    'encoder_hidden_dim': args.encoder_hidden_dim,
    'encoder_hidden_ff': args.encoder_hidden_ff,
    'patch_shape': args.patch_shape,
    'decoder_hidden_channels': args.decoder_hidden_channels,
    'train_kind': args.train_kind,
    'val_kind': args.val_kind,
}
np.save(run_dir / "config.npy", config_dict)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

init_conds, trajs = load_navier_stokes_tensor(args.data, n_timesteps=args.n_timesteps)
init_conds = init_conds.to(device)
trajs = trajs.to(device)

train_loader, val_loader = setup_dataloaders(init_conds, trajs, batch_size=args.batch_size)
P = 3
N, T, H, W, Q = trajs.shape

# %%
def make_block_mask_after(n_tokens, block_size):
    idx = torch.arange(n_tokens, dtype=torch.long)
    mask_after = block_size * ((idx // block_size) + 1) - 1
    return mask_after

# %%
from architectures import TrimTransformer, PatchwiseMLP, TimestepwiseMLP, PositionalEncoding, PositionalUnencoding
from kind_wrappers import AcausalWrapper, OneStepWrapper, GenerateWrapper
from architecture_wrappers import PicardIterations, ArbitraryIterations, make_weight_shared_modules, make_weight_unshared_modules
from training import Pipeline
from functools import partial

# %%
# Define the architecture

encoder_hidden_dim = args.encoder_hidden_dim if args.encoder_hidden_dim is not None else args.d_model - P

encoder = PatchwiseMLP(dim=Q,
                       hidden_dim=encoder_hidden_dim,
                       out_dim=args.d_model-P,
                       hidden_ff=args.encoder_hidden_ff,
                       K=args.patch_shape,
                       S=args.patch_shape)
encoder = encoder.to(device)

# Dummy forward pass to get shapes
with torch.no_grad():
    _, _, Hp, Wp, _ = encoder.forward(trajs[0, None, ...].to(device)).shape

decoder = TimestepwiseMLP(in_shape=torch.Size([Hp, Wp, args.d_model-P]),
                         layer_sizes=args.decoder_hidden_channels,
                         out_shape=torch.Size([H, W, Q]))

if args.train_kind == "acausal":
    assert args.val_kind == "acausal"
    time_width = args.n_timesteps + 1
else:
    time_width = args.n_timesteps
n_tokens = args.n_timesteps * Hp * Wp

pos_enc = PositionalEncoding(time_width, Hp, Wp)
pos_unenc = PositionalUnencoding(time_width, Hp, Wp)

scale = 1 / n_tokens
make_module = partial(TrimTransformer,
                      d_model=args.d_model,
                      nhead=args.nhead,
                      dim_feedforward=args.dim_feedforward,
                      dropout=args.dropout,
                      n_layers=args.n_layers,
                      scale=scale)

if args.share:
    modules = make_weight_shared_modules(make_module, n_modules=args.n_modules)
else:
    modules = make_weight_unshared_modules(make_module, n_modules=args.n_modules)

if args.picard:
    model = PicardIterations(modules, q=Q, r=args.r)
else:
    model = ArbitraryIterations(modules)

# %%
# Use a wrapper to handle the masking and whether or not to use the kv cache.

patch_size = Hp * Wp
mask = make_block_mask_after(n_tokens, patch_size).to(device)
acausal_model = AcausalWrapper(model)
one_step_model = OneStepWrapper(model, mask=mask)
generate_model = GenerateWrapper(model)

# Wrap all the components into a pipeline.
acausal_pipeline = Pipeline(acausal_model, encoder, decoder, pos_enc, pos_unenc)
one_step_pipeline = Pipeline(one_step_model, encoder, decoder, pos_enc, pos_unenc)
generate_pipeline = Pipeline(generate_model, encoder, decoder, pos_enc, pos_unenc)
acausal_pipeline.to(device)
one_step_pipeline.to(device)
generate_pipeline.to(device)

pipelines = {"acausal": acausal_pipeline, "one_step": one_step_pipeline, "generate": generate_pipeline}
train_pipeline = pipelines[args.train_kind]
val_pipeline = pipelines[args.val_kind]

# %%
loss_fn = F.mse_loss
optim = torch.optim.Adam(
    list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.T_max, eta_min=args.min_lr)

train_losses = []
val_losses = []
epoch_times = []

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()

    train_pipeline.train()
    train_loss = training_epoch(train_loader, train_pipeline, args.train_kind, loss_fn, optim)
    val_pipeline.eval()
    with torch.no_grad():
        val_loss   = evaluation_epoch(val_loader, val_pipeline, args.val_kind, loss_fn)
    scheduler.step()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    # Append losses and times to tracking lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_times.append(epoch_time)
    
    print(f"{args.name} | Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f} | time: {epoch_time:.2f}s")
    
    # Save losses and times as numpy arrays every epoch in run directory
    np.save(run_dir / "train_loss.npy", np.array(train_losses))
    np.save(run_dir / "val_loss.npy", np.array(val_losses))
    np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

# Save model weights in run directory
torch.save({"state_dict": model.state_dict()}, run_dir / "model_weights.pt")

# Save final loss arrays and times in run directory
np.save(run_dir / "train_loss.npy", np.array(train_losses))
np.save(run_dir / "val_loss.npy", np.array(val_losses))
np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

print(f"\nTraining completed! All files saved to: {run_dir}")
print(f"Final train loss: {train_losses[-1]:.6f}")
print(f"Final val loss: {val_losses[-1]:.6f}")
print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
print(f"Total training time: {np.sum(epoch_times):.2f}s")

# %%



