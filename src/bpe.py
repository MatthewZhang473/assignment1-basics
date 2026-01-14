import heapq
import os
import regex as re
import cProfile
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

# Note: Assuming find_chunk_boundaries is available in your environment
# If not, ensure it's imported from your local utilities.
from cs336_basics import find_chunk_boundaries


def pretokenize_chunk(args):
    chunk_text, pattern, show_progress = args
    token_counts = defaultdict(int)
    for m in tqdm(
        re.finditer(pattern, chunk_text),
        desc="Pretokenizing chunk",
        disable=not show_progress,
    ):
        b = m.group().encode("utf-8")
        # We store words as tuples of single-byte bytes objects
        token_counts[tuple(bytes([x]) for x in b)] += 1
    return token_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens=["<|endoftext|>"],
    debug=False,
    num_processes=4,
):
    # 1. INITIALIZATION
    pattern = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    vocab = [bytes([i]) for i in range(256)]
    special_bytes = [st.encode("utf8") for st in special_tokens]
    vocab += special_bytes

    # 2. PARALLEL PRETOKENIZATION
    token_counts_map = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")
            pattern_specials = "|".join(re.escape(st) for st in special_tokens)
            segments = re.split(pattern_specials, chunk)
            chunks.extend([seg for seg in segments if seg.strip()])

        chunk_args = [(chunk, pattern, i == 0) for i, chunk in enumerate(chunks)]
        with Pool(num_processes) as pool:
            chunk_results = pool.map(pretokenize_chunk, chunk_args)

        for chunk_counts in chunk_results:
            for token, count in chunk_counts.items():
                token_counts_map[token] += count

    # 3. BUILD OPTIMIZED STRUCTURES
    # We represent words as mutable lists of byte-tokens for fast merging
    words = []
    word_counts = []
    pair_freqs = defaultdict(int)
    # The Inverted Index: pair -> set of word indices containing that pair
    pair_to_word_indices = defaultdict(set)

    for word_tuple, count in token_counts_map.items():
        word_idx = len(words)
        word_list = list(word_tuple)
        words.append(word_list)
        word_counts.append(count)

        for i in range(len(word_list) - 1):
            pair = (word_list[i], word_list[i + 1])
            pair_freqs[pair] += count
            pair_to_word_indices[pair].add(word_idx)

    # 4. INITIALIZE PRIORITY QUEUE
    # heapq is a min-heap, so we use negative frequency for max-heap behavior
    hq = [(-freq, pair) for pair, freq in pair_freqs.items()]
    heapq.heapify(hq)

    merges = []

    # 5. OPTIMIZED MERGE LOOP
    progress_bar = tqdm(total=vocab_size - len(vocab), desc="Merging tokens")
    while len(vocab) < vocab_size:
        if not hq:
            break

        neg_freq, pair = heapq.heappop(hq)
        # Check if the frequency in the heap is stale (Lazy update)
        if -neg_freq != pair_freqs.get(pair, 0):
            continue

        if -neg_freq <= 0:
            break

        left, right = pair
        merged_token = left + right
        merges.append(pair)
        vocab.append(merged_token)

        # Surgical update: only look at words containing this pair
        affected_word_indices = list(pair_to_word_indices[pair])
        # Clean up as we go
        del pair_to_word_indices[pair]
        pair_freqs[pair] = 0

        for word_idx in affected_word_indices:
            word = words[word_idx]
            count = word_counts[word_idx]

            i = 0
            while i < len(word) - 1:
                if word[i] == left and word[i + 1] == right:
                    # Found a match! Update neighbor pairs before the merge
                    # Decrease frequency of the 'broken' neighbor pairs
                    if i > 0:
                        old_left_pair = (word[i - 1], word[i])
                        pair_freqs[old_left_pair] -= count
                    if i < len(word) - 2:
                        old_right_pair = (word[i + 1], word[i + 2])
                        pair_freqs[old_right_pair] -= count

                    # Merge the elements in the list
                    word[i : i + 2] = [merged_token]

                    # Increase frequency of the 'new' neighbor pairs
                    if i > 0:
                        new_left_pair = (word[i - 1], word[i])
                        pair_freqs[new_left_pair] += count
                        pair_to_word_indices[new_left_pair].add(word_idx)
                        heapq.heappush(hq, (-pair_freqs[new_left_pair], new_left_pair))
                    if i < len(word) - 1:
                        new_right_pair = (word[i], word[i + 1])
                        pair_freqs[new_right_pair] += count
                        pair_to_word_indices[new_right_pair].add(word_idx)
                        heapq.heappush(
                            hq, (-pair_freqs[new_right_pair], new_right_pair)
                        )
                else:
                    i += 1

        progress_bar.update(1)

    progress_bar.close()
    vocab_dict = {i: token for i, token in enumerate(vocab)}
    return vocab_dict, merges


def main():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    # Ensure profiling dir exists
    os.makedirs("profiling", exist_ok=True)

    num_new_merges = 10000
    vocab, merges = train_bpe(
        input_path=file_path,
        vocab_size=256 + num_new_merges,
        special_tokens=["<|endoftext|>"],
    )

    print("\nTop 10 Merged Tokens:")
    for i in range(256, 256 + 10):
        if i in vocab:
            print(f"ID {i}: {vocab[i]}")


if __name__ == "__main__":
    cProfile.run("main()", "profiling/bpe.prof")
