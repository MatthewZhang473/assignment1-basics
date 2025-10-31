from collections import defaultdict
import regex as re
import heapq
from multiprocessing import Pool
from tqdm import tqdm
from cs336_basics import find_chunk_boundaries


# ------------------------------------------------------------
#  Parallel pretokenization
# ------------------------------------------------------------
def pretokenize_chunk(args):
    chunk_text, pattern, show_progress = args
    token_counts = defaultdict(int)

    iterator = re.finditer(pattern, chunk_text)
    if show_progress:
        iterator = tqdm(iterator, desc="Pretokenizing chunk")

    for m in iterator:
        b = m.group().encode("utf-8")
        token_counts[tuple(bytes([x]) for x in b)] += 1
    return token_counts


# ------------------------------------------------------------
#  Helper functions for heap-based BPE
# ------------------------------------------------------------
def pair_key_for_heap(pair):
    # reverse lexicographic priority (bigger bytes sequence wins)
    return tuple(-x for b in pair for x in b)

def build_initial_structs(token_counts):
    tokens, tok_count = [], []
    for t, c in token_counts.items():
        tokens.append(list(t))
        tok_count.append(c)

    occ = defaultdict(dict)
    pair_freq = defaultdict(int)
    for tid, syms in enumerate(tokens):
        c = tok_count[tid]
        for i in range(len(syms) - 1):
            pair = (syms[i], syms[i + 1])
            occ[pair][(tid, i)] = occ[pair].get((tid, i), 0) + c
            pair_freq[pair] += c

    heap, uid = [], 0
    for p, cnt in pair_freq.items():
        if cnt > 0:
            heapq.heappush(heap, (-cnt, pair_key_for_heap(p), uid, p))
            uid += 1
    return tokens, tok_count, occ, pair_freq, heap, uid


def push_heap(heap, uid, pair, pair_freq):
    cnt = pair_freq.get(pair, 0)
    if cnt > 0:
        heapq.heappush(heap, (-cnt, pair_key_for_heap(pair), uid, pair))
        uid += 1
    return uid


def remove_occurrence(occ, pair, key, weight, pair_freq):
    if key in occ[pair]:
        pair_freq[pair] -= weight
        del occ[pair][key]
        if pair_freq[pair] <= 0:
            pair_freq[pair] = 0


def add_occurrence(occ, pair, key, weight, pair_freq):
    if weight <= 0:
        return
    prev = occ[pair].get(key, 0)
    occ[pair][key] = prev + weight
    pair_freq[pair] += weight


def heap_bpe(token_counts, base_vocab, vocab_size, debug=False):
    tokens, tok_count, occ, pair_freq, heap, uid = build_initial_structs(token_counts)
    vocab = list(base_vocab)
    merges = []

    while len(vocab) < vocab_size and heap:
        neg_cnt, _, _, pair = heapq.heappop(heap)
        cnt = -neg_cnt
        if pair_freq.get(pair, 0) != cnt or cnt == 0:
            continue  # stale heap entry

        L, R = pair
        M = L + R
        merges.append(pair)
        vocab.append(M)

        hits = sorted(occ[pair].items(), key=lambda kv: (kv[0][0], kv[0][1]))
        by_tok = defaultdict(list)
        for (tid, pos), w in hits:
            by_tok[tid].append((pos, w))

        for tid, entries in by_tok.items():
            syms = tokens[tid]
            c = tok_count[tid]
            consumed = set()
            entries.sort()
            shift = 0
            for pos, w in entries:
                pos -= shift
                if pos < 0 or pos + 1 >= len(syms):
                    continue
                if pos in consumed or (pos + 1) in consumed:
                    continue
                if not (syms[pos] == L and syms[pos + 1] == R):
                    continue

                prev_sym = syms[pos - 1] if pos - 1 >= 0 else None
                next_sym = syms[pos + 2] if pos + 2 < len(syms) else None

                remove_occurrence(occ, (L, R), (tid, pos), c, pair_freq)
                if prev_sym is not None:
                    remove_occurrence(occ, (prev_sym, L), (tid, pos - 1), c, pair_freq)
                if next_sym is not None:
                    remove_occurrence(occ, (R, next_sym), (tid, pos + 1), c, pair_freq)

                syms[pos:pos + 2] = [M]
                consumed.update({pos, pos + 1})
                shift += 1

                if prev_sym is not None:
                    add_occurrence(occ, (prev_sym, M), (tid, pos - 1), c, pair_freq)
                    uid = push_heap(heap, uid, (prev_sym, M), pair_freq)
                if next_sym is not None:
                    add_occurrence(occ, (M, next_sym), (tid, pos), c, pair_freq)
                    uid = push_heap(heap, uid, (M, next_sym), pair_freq)

    return vocab, merges


# ------------------------------------------------------------
#  Full training pipeline
# ------------------------------------------------------------
def train_bpe(input_path: str, vocab_size: int, special_tokens = ["<|endoftext|>"], debug=False, num_processes=4):
    """
    Trains a byte-pair encoding (BPE) tokenizer on `input_path` until `vocab_size` is reached.
    Returns:
        vocab (dict[int, bytes])
        merges (list[tuple[bytes, bytes]])
    """
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab = [bytes([i]) for i in range(256)]
    special_tokens = [st.encode('utf8') for st in special_tokens]
    vocab += special_tokens

    token_counts = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="replace")
            # Split the chunk on any special token so merges can't cross them
            pattern_specials = "|".join(re.escape(st.decode("utf-8")) for st in special_tokens)
            segments = re.split(pattern_specials, chunk)
            # Filter out empty strings
            segments = [seg for seg in segments if seg.strip()]
            chunks.extend(segments)

        chunk_args = [(chunk, pattern, i == 0) for i, chunk in enumerate(chunks)]
        with Pool(num_processes) as pool:
            chunk_results = pool.map(pretokenize_chunk, chunk_args)

        for chunk_counts in chunk_results:
            for token, count in chunk_counts.items():
                token_counts[token] += count

    vocab, merges = heap_bpe(token_counts, vocab, vocab_size, debug)
    vocab = {i: token for i, token in enumerate(vocab)}
    return vocab, merges


# ------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"
    file_path = "data/debug.txt"
    N = 18
    vocab, merges = train_bpe(input_path=file_path, vocab_size=256 + N, special_tokens=["<|endoftext|>"])
    # print([vocab[i] for i in range(256, 256 + N)])
    
    print(merges)
