import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.loss import cross_entropy
from src.training.optimizer import AdamW
from src.training_loop.checkpointing import load_checkpoint, save_checkpoint
from src.training_loop.data_loading import get_batch
from src.transformer.transformer_lm import TransformerLM


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Minimal training loop for assignment1")

    # Data
    parser.add_argument("--train_data", type=str, default="artifacts/tokenized/tinystories_train_tokens.bin")
    parser.add_argument("--val_data", type=str, default="artifacts/tokenized/tinystories_val_tokens.bin")
    parser.add_argument("--memmap_dtype", type=str, default="uint16")

    # Model
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1280)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--theta", type=float, default=10000.0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_iters", type=int, default=3500)

    # Logging / eval / checkpoint
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="artifacts/checkpoints/minimal_ckpt.pt")
    parser.add_argument("--resume", action="store_true")

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)

    # TensorBoard
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--tb_logdir", type=str, default="artifacts/tensorboard/training_together")

    return parser.parse_args(argv)


def estimate_split_loss(model, split_tokens, eval_batches, cfg):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(split_tokens, cfg.batch_size, cfg.context_length, cfg.device)
            logits = model(x)  # (B, T, V)
            bsz, seq_len, vocab = logits.shape
            loss = cross_entropy(logits.view(bsz * seq_len, vocab), y.view(bsz * seq_len))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main(argv=None):
    cfg = parse_config(argv)
    train_data_path = resolve_path(cfg.train_data)
    val_data_path = resolve_path(cfg.val_data)
    checkpoint_path = resolve_path(cfg.checkpoint_path)
    tb_logdir = resolve_path(cfg.tb_logdir)

    print(cfg)
    print(f"Resolved train_data: {train_data_path}")
    print(f"Resolved val_data:   {val_data_path}")
    print(f"Resolved checkpoint: {checkpoint_path}")
    if cfg.use_tensorboard:
        print(f"Resolved tb_logdir:  {tb_logdir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Fallback: if tokenized memmaps are missing, train directly on raw TinyStories bytes.
    active_train_path = train_data_path
    active_val_path = val_data_path
    active_dtype = cfg.memmap_dtype
    active_vocab_size = cfg.vocab_size

    if not train_data_path.exists() or not val_data_path.exists():
        raw_train = REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
        raw_val = REPO_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
        if raw_train.exists() and raw_val.exists():
            print(
                "Tokenized memmaps not found; falling back to raw TinyStories byte-level training "
                "(dtype=uint8, vocab_size=256)."
            )
            active_train_path = raw_train
            active_val_path = raw_val
            active_dtype = "uint8"
            active_vocab_size = 256
        else:
            raise FileNotFoundError(
                "Data files not found. Provide tokenized --train_data/--val_data, or ensure "
                "data/TinyStoriesV2-GPT4-train.txt and data/TinyStoriesV2-GPT4-valid.txt exist."
            )

    dtype = np.dtype(active_dtype)
    train_tokens = np.memmap(active_train_path, mode="r", dtype=dtype)
    val_tokens = np.memmap(active_val_path, mode="r", dtype=dtype)

    if len(train_tokens) <= cfg.context_length + 1 or len(val_tokens) <= cfg.context_length + 1:
        raise ValueError("Dataset is too small for the selected --context_length.")

    print(f"Loaded train memmap: {active_train_path} ({len(train_tokens):,} tokens)")
    print(f"Loaded val memmap:   {active_val_path} ({len(val_tokens):,} tokens)")

    device = torch.device(cfg.device)
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=active_vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        theta=cfg.theta,
        device=device,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )

    start_iter = 0
    if cfg.resume and checkpoint_path.exists():
        start_iter = load_checkpoint(str(checkpoint_path), model, optimizer)
        print(f"Resumed from {checkpoint_path} at iteration {start_iter}")

    tb_writer = None
    if cfg.use_tensorboard:
        tb_logdir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_logdir))
        print(f"TensorBoard logging to: {tb_logdir}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    last_time = time.time()

    for it in range(start_iter, cfg.max_iters):
        x, y = get_batch(train_tokens, cfg.batch_size, cfg.context_length, cfg.device)

        logits = model(x)  # (B, T, V)
        bsz, seq_len, vocab = logits.shape
        loss = cross_entropy(logits.view(bsz * seq_len, vocab), y.view(bsz * seq_len))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (it + 1) % cfg.log_interval == 0:
            now = time.time()
            dt = now - last_time
            last_time = now
            print(f"iter {it + 1:6d} | train_loss {loss.item():.4f} | dt {dt:.2f}s")
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss.item(), it + 1)

        if (it + 1) % cfg.eval_interval == 0:
            train_eval = estimate_split_loss(model, train_tokens, cfg.eval_batches, cfg)
            val_eval = estimate_split_loss(model, val_tokens, cfg.eval_batches, cfg)
            print(f"iter {it + 1:6d} | train_eval {train_eval:.4f} | val_eval {val_eval:.4f}")
            if tb_writer is not None:
                tb_writer.add_scalar("eval/train_loss", train_eval, it + 1)
                tb_writer.add_scalar("eval/val_loss", val_eval, it + 1)

        if (it + 1) % cfg.checkpoint_interval == 0 or (it + 1) == cfg.max_iters:
            save_checkpoint(model, optimizer, it + 1, str(checkpoint_path))
            print(f"saved checkpoint: {checkpoint_path} @ iter {it + 1}")

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
