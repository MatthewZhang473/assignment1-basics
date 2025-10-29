from collections import defaultdict
import regex as re
from pprint import pprint
from cs336_basics import find_chunk_boundaries
from multiprocessing import Pool
from tqdm import tqdm


def pretokenize_chunk(args):
    """
    Pretokenize a single chunk and return token counts.

    Args:
        args: tuple of (chunk_text, pattern, show_progress)

    Returns:
        dict: token counts for this chunk
    """
    chunk_text, pattern, show_progress = args
    token_counts = defaultdict(int)
    
    matches = list(re.finditer(pattern, chunk_text))
    iterator = tqdm(matches, desc="Pretokenizing chunk") if show_progress else matches
    
    for m in iterator:
        b = m.group().encode("utf-8")
        token_counts[tuple(bytes([x]) for x in b)] += 1
    return token_counts


def train_bpe(filepath: str, target_vocab_size: int, debug=False, num_processes=4):
    """
    Trains a byte-pair encoding (BPE) tokenizer on `text` until `target_vocab_size` is reached.

    Returns:
        vocab (dict[int, bytes]): Final vocabulary mapping token IDs to byte tokens.
        merges (list[tuple[bytes, bytes]]): List of BPE merges in creation order.
    """

    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = [bytes([i]) for i in range(256)]
    special_token = b"<|endoftext|>"
    special_token_str = special_token.decode("utf-8")
    vocab.append(special_token)

    # Parallel pretokenization
    token_counts = defaultdict(int)
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # Read all chunks
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace(special_token_str, "")
            chunks.append(chunk)
            if debug:
                print(chunk[0:100])
        
        # Prepare arguments with progress flag for first chunk only
        chunk_args = [
            (chunk, pattern, i == 0) 
            for i, chunk in enumerate(chunks)
        ]

        # Process chunks in parallel
        with Pool(num_processes) as pool:
            chunk_results = pool.map(pretokenize_chunk, chunk_args)

        # Merge results from all chunks
        for chunk_counts in chunk_results:
            for token, count in chunk_counts.items():
                token_counts[token] += count

    if debug:
        print("Initial pretokens:")
        pprint(token_counts)

    # initial pair frequencies
    pair_freqs = defaultdict(int)
    for token, count in token_counts.items():
        for i in range(len(token) - 1):
            pair_freqs[(token[i], token[i + 1])] += count
    if debug:
        print("\nInitial pair frequencies:")
        pprint(pair_freqs)

    ##### Don't change the code below #####

    def merge_pretoken(tokens: tuple, pair: tuple) -> tuple:
        merged = b''.join(pair)
        result, i, n = [], 0, len(pair)
        while i < len(tokens):
            if tokens[i:i+n] == pair:
                result.append(merged)
                i += n
            else:
                result.append(tokens[i])
                i += 1
        return tuple(result)

    merges = []
    while len(vocab) < target_vocab_size:
        if not pair_freqs:
            print("No more pairs to merge")
            break

        new_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
        left, right = new_pair
        merged_token = left + right
        if debug:
            print(f"\nMerging: {new_pair} -> {merged_token}")
        merges.append(new_pair)
        vocab.append(merged_token)
        del pair_freqs[new_pair]

        pretoken_updates = {}
        for token, count in token_counts.items():
            local_changes = defaultdict(int)
            has_merge = False

            for i in range(len(token) - 1):
                if (token[i], token[i + 1]) == new_pair:
                    has_merge = True
                    if i >= 1:
                        local_changes[(token[i - 1], token[i])] -= count
                        local_changes[(token[i - 1], merged_token)] += count
                    if i < len(token) - 2:
                        local_changes[(token[i + 1], token[i + 2])] -= count
                        local_changes[(merged_token, token[i + 2])] += count

            if has_merge:
                for k, v in local_changes.items():
                    pair_freqs[k] += v
                    if pair_freqs[k] <= 0:
                        del pair_freqs[k]
                new_token = merge_pretoken(token, (left, right))
                pretoken_updates[token] = (new_token, count)

        for old, (new, c) in pretoken_updates.items():
            del token_counts[old]
            token_counts[new] = c
        if debug:
            print(token_counts)

    vocab = {i: token for i, token in enumerate(vocab)}
    return vocab, merges


if __name__ == "__main__":
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    N = 200
    vocab, merges = train_bpe(filepath=file_path, target_vocab_size=256+N)
    print([vocab[i] for i in range(256, 256+N)])