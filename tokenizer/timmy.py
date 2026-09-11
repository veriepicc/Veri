import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# timmy v2, veri byte level bpe tokenizer
# production grade, trained on fineweb-edu + veri sft
# v2 fixes: no nfkc mangling code, single digit numbers for math,
# byte fallback on, wider code alphabet

TIMMY_VERSION = "2.0"

TIMMY_SPECIALS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<user>",
    "<assistant>",
    "<system>",
    "<think>",
    "</think>",
    "<code>",
    "</code>",
    "<reasoning>",
    "<answer>",
    "<tool_call>",
    "<sep>",
]

TIMMY_VOCAB_SIZE = 65536

# full byte alphabet so every byte incl newline and tab survives
# note: must be bytelevel mapped chars like G for space, not raw chars
# or decode emits unk on newlines
TIMMY_ALPHABET = ByteLevel.alphabet()


class Timmy:
    def __init__(self, path=None, max_len=None):
        self.tok = None
        self.max_len = max_len
        if path is not None:
            self.load(path)

    def train(self, iterator, vocab_size=TIMMY_VOCAB_SIZE, min_freq=2, show_progress=True):
        # v2 stack, order matters here
        # digits split forces single digit numbers so math generalizes
        # bytelevel handles the rest incl spaces, decode inverts exactly
        # note: no whitespacesplit, its decoder rejoins pieces with
        # spaces and corrupts text like <user>hello
        base = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
        base.normalizer = None
        base.pre_tokenizer = Sequence(
            [
                Digits(individual_digits=True),
                ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        base.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=TIMMY_SPECIALS,
            show_progress=show_progress,
            initial_alphabet=TIMMY_ALPHABET,
        )
        base.train_from_iterator(iterator, trainer=trainer)
        self.tok = base
        return self

    def encode(self, text, add_bos=False, add_eos=False, max_len=None):
        # returns list of ids
        if self.tok is None:
            raise ValueError("timmy not loaded, call load or train first")
        if not isinstance(text, str):
            raise TypeError(f"encode wants str, got {type(text).__name__}")
        s = text
        if add_bos:
            s = "<bos>" + s
        if add_eos:
            s = s + "<eos>"
        ids = self.tok.encode(s).ids
        lim = max_len if max_len is not None else self.max_len
        if lim is not None and len(ids) > lim:
            ids = ids[:lim]
        return ids

    def encode_batch(self, texts, add_bos=False, add_eos=False, max_len=None):
        # batch version, same order as input
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos, max_len=max_len) for t in texts]

    def decode(self, ids, skip_special=False):
        # turns ids back to text
        if self.tok is None:
            raise ValueError("timmy not loaded, call load or train first")
        return self.tok.decode(list(ids), skip_special_tokens=skip_special)

    def decode_batch(self, batch, skip_special=False):
        return [self.decode(ids, skip_special=skip_special) for ids in batch]

    def __len__(self):
        if self.tok is None:
            raise ValueError("timmy not loaded, call load or train first")
        return self.tok.get_vocab_size()

    def token_to_id(self, token):
        return self.tok.token_to_id(token)

    def id_to_token(self, idx):
        return self.tok.id_to_token(idx)

    def get_vocab(self):
        return self.tok.get_vocab()

    def enable_padding(self, pad_id=None, pad_token="<pad>"):
        # wire padding for batched training loops
        from tokenizers.processors import TemplateProcessing

        pid = pad_id if pad_id is not None else self.token_to_id(pad_token)
        self.tok.enable_padding(pad_id=pid, pad_token=pad_token)
        return self

    def enable_truncation(self, max_length):
        self.tok.enable_truncation(max_length=max_length)
        return self

    def pad_batch(self, batch, pad_id=None):
        # pads list of id lists to equal len, returns tensor-ready lists + mask
        pid = pad_id if pad_id is not None else self.token_to_id("<pad>")
        if pid is None:
            raise ValueError("no pad token in vocab, train with <pad> special first")
        n = max(len(x) for x in batch)
        ids = [list(x) + [pid] * (n - len(x)) for x in batch]
        mask = [[1] * len(x) + [0] * (n - len(x)) for x in batch]
        return ids, mask

    def save(self, path):
        # writes timmy json + config sidecar
        p = Path(path)
        self.tok.save(str(p))
        cfg = {
            "name": "timmy",
            "version": TIMMY_VERSION,
            "vocab_size": self.tok.get_vocab_size(),
            "specials": TIMMY_SPECIALS,
            "type": "bpe_bytelevel_v2",
        }
        p.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return str(p)

    def load(self, path):
        # loads from timmy json
        self.tok = Tokenizer.from_file(str(path))
        return self

    def vocab_size(self):
        return self.tok.get_vocab_size()

    def chars_per_token(self, texts):
        # compression metric, higher is better
        chars = sum(len(t) for t in texts)
        toks = sum(len(self.encode(t)) for t in texts)
        return chars / max(1, toks)

    def apply_chat(self, messages, add_generation_prompt=False):
        # messages is list of dicts with role + content
        # roles are system user assistant, unknown roles raise
        # keep think tags raw so reasoning stays cheap
        out = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                out += "<system>" + content + "<eos>"
            elif role == "user":
                out += "<user>" + content + "<eos>"
            elif role == "assistant":
                out += "<assistant>" + content + "<eos>"
            else:
                raise ValueError(f"unknown chat role: {role!r}, want system user assistant")
        if add_generation_prompt:
            out += "<assistant>"
        return out

    def encode_chat(self, messages, add_generation_prompt=False, max_len=None):
        return self.encode(self.apply_chat(messages, add_generation_prompt), max_len=max_len)

    def save_pretrained(self, directory):
        # hf hub layout so transformers can load it later
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(d / "tokenizer.json"))
        cfg = {
            "name": "timmy",
            "version": TIMMY_VERSION,
            "vocab_size": self.tok.get_vocab_size(),
            "specials": TIMMY_SPECIALS,
            "type": "bpe_bytelevel_v2",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
        }
        (d / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return str(d)
