import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from timmy import Timmy, TIMMY_SPECIALS  # noqa: E402


# tiny deterministic corpus, no network needed
CORPUS = (
    ["hello veri reasoning trace number 12345 <think>step by step</think>"] * 300
    + ["def rmsnorm(x, w, eps=1e-6):\n    return x / 2"] * 300
    + ["<user>who are you?</user><assistant><think>veri built from scratch</think>veri.</assistant>"] * 300
    + ["price 80235 dollars, 67890 cents, total 148125"] * 300
)

ROUNDTRIPS = [
    "hello veri",
    "<user>hi</user><assistant><think>yo</think>yo</assistant>",
    "def f(x):\n\treturn x * 2\n\n\ndone",
    "12345 + 67890 = 80235",
    "unicode caf\u00e9 \u4e2d\u6587 \U0001f600 ok",
    "",
]


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def main():
    t = Timmy()
    t.train(iter(CORPUS), vocab_size=4000, min_freq=2, show_progress=False)
    # small corpus runs out of pairs early, trainer stops instead of junk merges
    check("vocab bounded", 300 < t.vocab_size() <= 4000)
    check("len matches vocab", len(t) == t.vocab_size())

    for s in ROUNDTRIPS:
        check(f"roundtrip {ascii(s)[:40]}", t.decode(t.encode(s)) == s)

    # specials are atomic
    for sp in ["<think>", "</think>", "<user>", "<assistant>", "<bos>", "<eos>", "<pad>"]:
        check(f"single token {sp}", len(t.encode(sp)) == 1)

    # digits split for math
    check("digits split", t.tok.encode("12345").tokens == ["1", "2", "3", "4", "5"])

    # batch keeps order
    texts = ["hello", "world veri", "<think>hi</think>"]
    batch = t.encode_batch(texts)
    check("batch order", [t.decode(x) for x in batch] == texts)
    check("decode batch", t.decode_batch(batch) == texts)

    # truncation
    long_ids = t.encode("hello veri " * 200, max_len=32)
    check("truncation", len(long_ids) == 32)

    # padding
    t.enable_padding()
    ids, mask = t.pad_batch([[1, 2, 3], [4, 5]])
    check("pad shape", len(ids) == 2 and len(ids[0]) == 3 and len(ids[1]) == 3)
    check("pad mask", mask == [[1, 1, 1], [1, 1, 0]])
    check("pad id is pad", ids[1][2] == t.token_to_id("<pad>"))

    # chat template
    msgs = [
        {"role": "system", "content": "you are veri"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>x</think>y"},
    ]
    s = t.apply_chat(msgs)
    check("chat roles", "<system>" in s and "<user>" in s and "<assistant>" in s)
    check("chat gen prompt", t.apply_chat(msgs[:2], True).endswith("<assistant>"))
    try:
        t.apply_chat([{"role": "tool", "content": "x"}])
        check("chat bad role raises", False)
    except ValueError:
        check("chat bad role raises", True)

    # type safety
    try:
        t.encode(123)
        check("encode type error", False)
    except TypeError:
        check("encode type error", True)

    # save load roundtrip
    with tempfile.TemporaryDirectory() as d:
        t.save(str(Path(d) / "timmy.json"))
        t2 = Timmy(str(Path(d) / "timmy.json"))
        check("save load", t2.decode(t2.encode("hello veri")) == "hello veri")
        t.save_pretrained(str(Path(d) / "hf"))
        check("hf layout", (Path(d) / "hf" / "tokenizer.json").exists())
        check("hf config", (Path(d) / "hf" / "tokenizer_config.json").exists())

    # specials all present
    v = t.get_vocab()
    for sp in TIMMY_SPECIALS:
        check(f"special in vocab {sp}", sp in v)

    print("ALL TIMMY TESTS PASS")


if __name__ == "__main__":
    main()
