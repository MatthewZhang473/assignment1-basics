from collections.abc import Iterable, Iterator
import pickle
import regex as re


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Assuming the text can fit in memory. Use encode_iterable otherwise."""

        # 1. Handle special tokens - split text by special tokens
        if self.special_tokens:
            pattern_specials = "|".join(re.escape(st) for st in self.special_tokens)
            segments = re.split(f"({pattern_specials})", text)
        else:
            segments = [text]

        all_tk_ids = []

        for segment in segments:
            # If this segment is a special token, encode it directly
            if self.special_tokens and segment in self.special_tokens:
                special_token_bytes = segment.encode("utf-8")
                all_tk_ids.append(self.inverse_vocab[special_token_bytes])
            elif segment:  # Non-empty regular text
                pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pretokens = [m.group() for m in re.finditer(pattern, segment)]

                # Encode pretokens
                tk_ids = self._encode_pretokens(pretokens=pretokens)
                all_tk_ids.extend(tk_ids)

        return all_tk_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            pattern_specials = "|".join(re.escape(st) for st in self.special_tokens)
        else:
            pattern_specials = None

        for chunk in iterable:  # each line
            segments = (
                re.split(f"({pattern_specials})", chunk)
                if pattern_specials
                else [chunk]
            )
            for segment in segments:
                if self.special_tokens and segment in self.special_tokens:
                    yield self.inverse_vocab[segment.encode("utf-8")]
                elif segment:
                    pretokens = [m.group() for m in re.finditer(PATTERN, segment)]
                    yield from self._encode_pretokens(pretokens)

    def decode(self, ids: list[int]) -> str:
        byte_stream = b"".join(self.vocab[i] for i in ids)
        return byte_stream.decode("utf-8", errors="replace")

    def _encode_pretokens(self, pretokens):

        def apply_merge(subwords, m):
            n = len(subwords)
            new_subwords = []
            combined = m[0] + m[1]
            i = 0
            while i < n - 1:
                if subwords[i] == m[0] and subwords[i + 1] == m[1]:
                    new_subwords.append(combined)
                    i += 2
                else:
                    new_subwords.append(subwords[i])
                    i += 1
            if i == n - 1:
                new_subwords.append(subwords[n - 1])

            # print(f"subwords: {subwords}")
            # print(f"new_subwords: {new_subwords}")
            return new_subwords

        all_tk_ids = []
        cache = {}
        # split all pretokens by bytes -> list(bytes)
        for pt in pretokens:
            if pt in cache:
                tk_ids = cache[pt]
            else:
                subwords = [bytes([b]) for b in pt.encode("utf-8")]
                for m in self.merges:
                    subwords = apply_merge(subwords, m)
                tk_ids = [self.inverse_vocab[sw] for sw in subwords]
                cache[pt] = tk_ids
            all_tk_ids += tk_ids

        return all_tk_ids


if __name__ == "__main__":

    vocab = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    text = "the cat ate"
    tkzr = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    all_tk_ids = tkzr.encode(text)
    print(all_tk_ids)

    print(tkzr.decode(all_tk_ids))
