import argparse
import json
import glob
from pathlib import Path

from timmy import Timmy, TIMMY_SPECIALS


# timmy trainer
# streams fineweb-edu sample + local veri sft jsonl, builds 64k byte level bpe
# run: python train.py --vocab 65536 --out timmy.json

SPECIAL = TIMMY_SPECIALS


def iter_fineweb_edu(sample="sample-100BT", limit=2_000_000):
    # lazy import so local runs without hf dont crash
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=sample,
        split="train",
        streaming=True,
    )
    n = 0
    for row in ds:
        text = row.get("text", "")
        if len(text) < 200:
            continue
        yield text
        n += 1
        if n >= limit:
            break


def iter_veri_sft(repeat=8):
    # approved corpus snapshot lives in Veri/datasets_clean
    # only clean files were copied there, nothing else exists to filter
    base = Path(__file__).resolve().parent.parent / "datasets_clean"
    files = []
    if base.exists():
        files += sorted(glob.glob(str(base / "**" / "*.jsonl"), recursive=True))

    # dedupe, keep small files only for tokenizer training
    seen = set()
    for _ in range(repeat):
        for fp in files:
            if fp in seen and repeat == 1:
                continue
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            yield line[:1500]
                            continue
                        # common shapes in veri dumps, capped, trainer holds all in ram
                        for key in ("text", "content", "output", "response", "completion"):
                            if key in obj and isinstance(obj[key], str) and len(obj[key]) > 20:
                                yield obj[key][:1500]
                                break
                        else:
                            # fallback dump whole row
                            yield json.dumps(obj, ensure_ascii=False)[:1500]
            except Exception:
                continue
        seen.update(files)
        if not files:
            break


def combined_iter(sample, limit, veri_repeat):
    yield from iter_veri_sft(repeat=veri_repeat)
    yield from iter_fineweb_edu(sample=sample, limit=limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--out", type=str, default="timmy.json")
    ap.add_argument("--sample", type=str, default="sample-100BT")
    ap.add_argument("--limit", type=int, default=500_000)
    ap.add_argument("--veri_repeat", type=int, default=8)
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    # show corpus contents so block list is verifiable
    from pathlib import Path as _P

    _root = _P(__file__).resolve().parent.parent / "datasets_clean"
    _names = sorted(fp.name for fp in _root.rglob("*.jsonl")) if _root.exists() else []
    print(f"corpus files ({len(_names)}): {_names}")

    t = Timmy()
    t.train(
        combined_iter(args.sample, args.limit, args.veri_repeat),
        vocab_size=args.vocab,
        min_freq=args.min_freq,
    )

    out = Path(__file__).resolve().parent / args.out
    t.save(str(out))
    print(f"saved to {out}")
    # quick sanity check
    ids = t.encode("hello veri <think>reasoning is cheap when merges are good</think>")
    print(f"tokens for test string: {len(ids)} -> {ids[:20]}")
    print(t.decode(ids))


if __name__ == "__main__":
    main()
