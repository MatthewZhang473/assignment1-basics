import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer.bpe import train_bpe
from src.tokenizer.tokenizer import Tokenizer


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train BPE tokenizer and tokenize TinyStories into memmap .bin files"
    )
    p.add_argument(
        "--tokenizer_train_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Corpus used to train BPE tokenizer.",
    )
    p.add_argument(
        "--tokenizer_train_chars",
        type=int,
        default=50_000_000,
        help="Train tokenizer on only this many characters from tokenizer_train_path.",
    )
    p.add_argument(
        "--train_text_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
    )
    p.add_argument(
        "--val_text_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
    )
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
    )
    p.add_argument("--num_processes", type=int, default=6)

    p.add_argument("--out_dir", type=str, default="artifacts/tokenized")
    p.add_argument("--vocab_out", type=str, default="tinystories_vocab.pkl")
    p.add_argument("--merges_out", type=str, default="tinystories_merges.pkl")
    p.add_argument(
        "--train_tokens_out", type=str, default="tinystories_train_tokens.bin"
    )
    p.add_argument("--val_tokens_out", type=str, default="tinystories_val_tokens.bin")
    p.add_argument(
        "--token_dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32", "int64"],
        help="Output dtype for token ID memmaps.",
    )

    return p.parse_args(argv)


def write_text_subset(src: Path, dst: Path, max_chars: int) -> int:
    written = 0
    with (
        src.open("r", encoding="utf-8", errors="replace") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        while written < max_chars:
            to_read = min(1_000_000, max_chars - written)
            chunk = fin.read(to_read)
            if not chunk:
                break
            fout.write(chunk)
            written += len(chunk)
    return written


def count_tokens(tokenizer: Tokenizer, text_path: Path, split_name: str) -> int:
    count = 0
    with text_path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in tqdm(
            tokenizer.encode_iterable(f),
            desc=f"Counting {split_name} tokens",
            unit="tok",
        ):
            count += 1
    return count


def write_tokens_memmap(
    tokenizer: Tokenizer,
    text_path: Path,
    out_path: Path,
    dtype: np.dtype,
    split_name: str,
):
    total = count_tokens(tokenizer, text_path, split_name)
    arr = np.memmap(out_path, mode="w+", dtype=dtype, shape=(total,))

    i = 0
    with text_path.open("r", encoding="utf-8", errors="replace") as f:
        for token_id in tqdm(
            tokenizer.encode_iterable(f),
            total=total,
            desc=f"Writing {split_name} tokens",
            unit="tok",
        ):
            arr[i] = token_id
            i += 1

    arr.flush()
    return total


def main(argv=None):
    args = parse_args(argv)

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_train_path = resolve_path(args.tokenizer_train_path)
    train_text_path = resolve_path(args.train_text_path)
    val_text_path = resolve_path(args.val_text_path)

    for p in [tokenizer_train_path, train_text_path, val_text_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    print("Training tokenizer...")
    print(f"tokenizer_train_path={tokenizer_train_path}")
    subset_path = out_dir / "tokenizer_train_subset.txt"
    subset_chars = write_text_subset(
        tokenizer_train_path, subset_path, args.tokenizer_train_chars
    )
    print(f"Created tokenizer subset: {subset_path} ({subset_chars:,} chars)")
    print(
        "Tokenizer training uses a subset of train split to avoid validation leakage."
    )

    vocab, merges = train_bpe(
        input_path=str(subset_path),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )

    vocab_path = out_dir / args.vocab_out
    merges_path = out_dir / args.merges_out
    with vocab_path.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_path.open("wb") as f:
        pickle.dump(merges, f)

    tokenizer = Tokenizer(
        vocab=vocab, merges=merges, special_tokens=args.special_tokens
    )

    dtype = np.dtype(args.token_dtype)
    max_id = max(vocab.keys())
    if np.issubdtype(dtype, np.unsignedinteger):
        max_allowed = np.iinfo(dtype).max
        if max_id > max_allowed:
            raise ValueError(
                f"token_dtype={args.token_dtype} cannot hold token IDs up to {max_id}. "
                f"Use a larger dtype (e.g., uint32)."
            )

    train_out = out_dir / args.train_tokens_out
    val_out = out_dir / args.val_tokens_out

    print(f"Tokenizing train split -> {train_out}")
    n_train = write_tokens_memmap(
        tokenizer, train_text_path, train_out, dtype, split_name="train"
    )
    print(f"train tokens: {n_train:,}")

    print(f"Tokenizing val split -> {val_out}")
    n_val = write_tokens_memmap(
        tokenizer, val_text_path, val_out, dtype, split_name="val"
    )
    print(f"val tokens: {n_val:,}")

    print("Done.")
    print(f"vocab_path={vocab_path}")
    print(f"merges_path={merges_path}")
    print(f"vocab_size={len(vocab)}")
    print("Use these in training:")
    print(
        "python scripts/train_together.py "
        f"--train_data {train_out} --val_data {val_out} "
        f"--memmap_dtype {args.token_dtype} --vocab_size {len(vocab)} --use_tensorboard"
    )


if __name__ == "__main__":
    main()
