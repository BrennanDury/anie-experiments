# %%
import torch
from torch.nn import functional as F
import numpy as np
import argparse
from pathlib import Path
from data import setup_dataloaders, load_navier_stokes_tensor
import time
from training import training_epoch, evaluation_epoch

parser = argparse.ArgumentParser(description="Navier–Stokes training script.")
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--data", type=Path, default=Path("ns_data.mat"), help="Path to the .mat dataset.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 penalty) for Adam optimizer.")
parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm (set to 0 to disable).")
parser.add_argument("--min-lr", type=float, default=1e-9, help="Minimum learning rate for cosine annealing scheduler.")
parser.add_argument("--T-max", type=int, default=101, help="Maximum number of iterations for cosine annealing scheduler.")
parser.add_argument("--start-factor", type=float, default=1e-5, help="Start factor for linear learning rate warmup.")
parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of epochs for linear learning rate warmup.")
parser.add_argument("--n-timesteps", type=int, default=10, help="Number of temporal frames to sample from the raw data (consistent with notebook).")

parser.add_argument("--share", action="store_true", help="Share weights between modules.")
parser.add_argument("--no-share", dest="share", action="store_false", help="Don't share weights between modules.")
parser.set_defaults(share=True)

parser.add_argument("--picard", action="store_true", help="Use Picard iterations.")
parser.add_argument("--no-picard", dest="picard", action="store_false", help="Don't use Picard iterations.")
parser.set_defaults(picard=True)

parser.add_argument("--project-input", action="store_true", help="Project input to model dimension")
parser.add_argument("--no-project-input", dest="project_input", action="store_false", help="Don't project input to model dimension")
parser.set_defaults(project_input=True)

parser.add_argument("--d-model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--dim-feedforward", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n-layers", type=int, default=4)
parser.add_argument("--n-modules", type=int, default=3)
parser.add_argument("--r", type=float, default=0.5)

# Encoder/Decoder arguments
parser.add_argument("--encoder-output-dim", type=int, default=32)
parser.add_argument("--encoder-channels", type=int, nargs="+", default=[32, 32],
                    help="Hidden channels for decoder MLP (excluding final output channel)")
parser.add_argument("--decoder-channels", type=int, nargs="+", default=[64, 256],
                    help="Hidden channels for decoder MLP (excluding final output channel)")
parser.add_argument("--decoder-kind", type=str, default="timestepwise", choices=["timestepwise", "patchwise"])
parser.add_argument("--patch-shape", type=int, nargs=2, default=[4, 4],
                    help="A token is a patch of size patch_shape")

parser.add_argument("--norm-mlp", action="store_true", help="Use layer normalization in MLP")
parser.add_argument("--no-norm-mlp", dest="norm-mlp", action="store_false", help="Don't use layer normalization in MLP")
parser.set_defaults(norm_mlp=False)

parser.add_argument("--train-kind", choices=["acausal", "one-step", "generate"], default="acausal",
                    help="Pipeline kind to use during training")
parser.add_argument("--val-kind", choices=["acausal", "one-step", "generate"], default="acausal",
                    help="Pipeline kind to use during validation")

parser.add_argument("--positional-encoding", type=str, choices=["coordinate", "rope", "learned"], default="coordinate")


args = parser.parse_args()

# %%
# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

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
    'start_factor': args.start_factor,
    'warmup_epochs': args.warmup_epochs,
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
    'project_input': args.project_input,
    'encoder_output_dim': args.encoder_output_dim,
    'patch_shape': args.patch_shape,
    'encoder_channels': args.encoder_channels,
    'decoder_channels': args.decoder_channels,
    'decoder_kind': args.decoder_kind,
    'norm_mlp': args.norm_mlp,
    'train_kind': args.train_kind,
    'val_kind': args.val_kind,
    'positional_encoding': args.positional_encoding,
}
np.save(run_dir / "config.npy", config_dict)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

init_conds, trajs = load_navier_stokes_tensor(args.data, n_timesteps=args.n_timesteps)
#init_conds, trajs = torch.randn(5000, 1, 64, 64, 1), torch.randn(5000, 10, 64, 64, 1)
init_conds = init_conds.to(device)
trajs = trajs.to(device)

train_loader, val_loader = setup_dataloaders(init_conds, trajs, batch_size=args.batch_size)
if args.positional_encoding == "coordinate":
    P = 3
else:
    P = 0
N, T, H, W, Q = trajs.shape

# %%
def make_block_mask_after(n_tokens, block_size):
    idx = torch.arange(n_tokens, dtype=torch.long)
    mask_after = block_size * ((idx // block_size) + 1) - 1
    return mask_after

# %%
from architectures import TrimTransformer, PatchwiseMLP, PatchwiseMLP_norm, TimestepwiseMLP, \
    PositionalEncoding, PositionalUnencoding, RoPENd, RoPENdUnencoding, \
    LearnedPositionalEncoding, LearnedPositionalUnencoding, DecoderWrapper
from kind_wrappers import AcausalWrapper, OneStepWrapper, GenerateWrapper
from architecture_wrappers import PicardIterations, ArbitraryIterations, \
    make_weight_shared_modules, make_weight_unshared_modules
from training import Pipeline
from functools import partial

# %%
# Define the architecture
if args.norm_mlp:
    MLP_class = PatchwiseMLP_norm
else:
    MLP_class = PatchwiseMLP

if args.project_input:
    transformer_input_dim = args.encoder_output_dim + P
else:
    transformer_input_dim = None
    assert args.d_model == args.encoder_output_dim + P


encoder = MLP_class(input_dim=Q,
                    output_dim=args.encoder_output_dim,
                    hidden_dims=args.encoder_channels,
                    K=args.patch_shape,
                    S=args.patch_shape)
encoder = encoder.to(device)

# Dummy forward pass to get shapes
with torch.no_grad():
    _, _, Hp, Wp, _ = encoder.forward(trajs[:1].to(device)).shape
patch_size = H*W // (Hp * Wp)

if args.decoder_kind == "timestepwise":
    decoder = TimestepwiseMLP(in_shape=torch.Size([Hp, Wp, args.encoder_output_dim]),
                             hidden_dims=args.decoder_channels,
                             out_shape=torch.Size([H, W, Q]))
else:
    decoder = DecoderWrapper(MLP_class(input_dim=args.encoder_output_dim,
                                       output_dim=Q*patch_size,
                                       hidden_dims=list(reversed(args.encoder_channels)),
                                       K=[1,1],
                                       S=[1,1]),
                                       Q=Q, patch_shape=args.patch_shape)

if args.train_kind == "acausal":
    assert args.val_kind == "acausal"
    time_width = args.n_timesteps + 1
else:
    time_width = args.n_timesteps
n_tokens = time_width * Hp * Wp

if args.positional_encoding == "coordinate":
    pos_enc = PositionalEncoding(time_width, Hp, Wp)
    pos_unenc = PositionalUnencoding(time_width, Hp, Wp)
elif args.positional_encoding == "rope":
    pos_enc = RoPENd(torch.Size([time_width, Hp, Wp,args.encoder_output_dim]))
    pos_unenc = RoPENdUnencoding(torch.Size([time_width, Hp, Wp, args.encoder_output_dim]))
else:
    pos_enc = LearnedPositionalEncoding(torch.Size([time_width, Hp, Wp, args.encoder_output_dim]))
    pos_unenc = LearnedPositionalUnencoding(pos_enc)

scale = 1 / n_tokens
make_module = partial(TrimTransformer,
                      d_model=args.d_model,
                      nhead=args.nhead,
                      dim_feedforward=args.dim_feedforward,
                      dropout=args.dropout,
                      n_layers=args.n_layers,
                      scale=scale,
                      input_dim=transformer_input_dim)

if args.share:
    modules = make_weight_shared_modules(make_module, n_modules=args.n_modules)
else:
    modules = make_weight_unshared_modules(make_module, n_modules=args.n_modules)

if args.picard:
    model = PicardIterations(modules, q=Q, r=args.r)
else:
    model = ArbitraryIterations(modules)

# %%
print("encoder params", sum(p.numel() for p in encoder.parameters()))
print("decoder params", sum(p.numel() for p in decoder.parameters()))
print("model params", sum(p.numel() for p in model.parameters()))

# %%
# Use a wrapper to handle the masking and whether or not to use the kv cache.
n_patches = Hp * Wp
mask = make_block_mask_after(n_tokens, n_patches).to(device)
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

pipelines = {"acausal": acausal_pipeline, "one-step": one_step_pipeline, "generate": generate_pipeline}
train_pipeline = pipelines[args.train_kind]
val_pipeline = pipelines[args.val_kind]

# %%
loss_fn = F.mse_loss
optim = torch.optim.Adam(
    pos_enc.parameters() if args.positional_encoding == "learned" else [] +
    list(model.parameters()) +
    list(encoder.parameters()) +
    list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay
)
if args.warmup_epochs > 0:
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=args.start_factor, total_iters=len(train_loader) * args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.T_max, eta_min=args.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[len(train_loader) * args.warmup_epochs])
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.T_max, eta_min=args.min_lr)

train_losses = []
val_losses = []
epoch_times = []

for epoch in range(1, args.epochs + 1):
    print(f"Epoch {epoch:3d} | lr: {optim.param_groups[0]['lr']:.6f}")
    epoch_start_time = time.time()

    train_pipeline.train()
    if epoch <= args.warmup_epochs:
        train_loss = training_epoch(train_loader, train_pipeline, args.train_kind, loss_fn, optim, scheduler=scheduler, grad_clip_norm=args.grad_clip_norm)
    else:
        train_loss = training_epoch(train_loader, train_pipeline, args.train_kind, loss_fn, optim, grad_clip_norm=args.grad_clip_norm)
    val_pipeline.eval()
    with torch.no_grad():
        val_loss   = evaluation_epoch(val_loader, val_pipeline, args.val_kind, loss_fn)
    if epoch > args.warmup_epochs:
        scheduler.step()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    epoch_times.append(epoch_time)

    print(f"{args.name} | Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f} | time: {epoch_time:.2f}s")
    
    np.save(run_dir / "train_loss.npy", np.array(train_losses))
    np.save(run_dir / "val_loss.npy", np.array(val_losses))
    np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

torch.save({"state_dict": model.state_dict()}, run_dir / "model_weights.pt")

np.save(run_dir / "train_loss.npy", np.array(train_losses))
np.save(run_dir / "val_loss.npy", np.array(val_losses))
np.save(run_dir / "epoch_times.npy", np.array(epoch_times))

print(f"\nTraining completed! All files saved to: {run_dir}")
print(f"Final train loss: {train_losses[-1]:.6f}")
print(f"Final val loss: {val_losses[-1]:.6f}")
print(f"Average epoch time: {np.mean(epoch_times):.2f}s")
print(f"Total training time: {np.sum(epoch_times):.2f}s")