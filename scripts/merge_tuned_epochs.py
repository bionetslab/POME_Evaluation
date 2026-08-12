"""Merge sharded POME tuned-epochs logs into one ``tuned_epochs.csv``.

``scripts/generate_inductive_embeddings.py`` rewrites its tuned-epochs log after
every computed run, so concurrent jobs must not share one file -- they each pass
a unique ``--log-tag`` and write ``tuned_epochs.<tag>.csv``. This script folds
those shards (plus any pre-existing ``tuned_epochs.csv``) back into a single
log, keyed by (dataset, mode, split, dim, run).

Run from the project root after a sharded submission has finished:

    python scripts/merge_tuned_epochs.py
    python scripts/merge_tuned_epochs.py --keep-shards --output-dir output/inductive

Later shards win on a duplicate key, and the shard files are deleted once the
merged log is written (use --keep-shards to retain them).
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "inductive"
KEY = ["dataset", "mode", "split", "dim", "run"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="inductive output root (default: output/inductive)")
    parser.add_argument("--keep-shards", action="store_true",
                        help="do not delete the tuned_epochs.<tag>.csv shards")
    args = parser.parse_args()

    pome_dir = args.output_dir / "pome"
    merged_path = pome_dir / "tuned_epochs.csv"
    shards = sorted(pome_dir.glob("tuned_epochs.*.csv"))
    if not shards:
        raise SystemExit(f"No tuned_epochs.<tag>.csv shards under {pome_dir}")

    frames = []
    # The merged file goes first so a shard's row wins on a duplicate key.
    if merged_path.exists():
        frames.append(pd.read_csv(merged_path))
        print(f"  existing {merged_path.name}: {len(frames[-1])} rows")
    for shard in shards:
        df = pd.read_csv(shard)
        frames.append(df)
        print(f"  {shard.name}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)
    # Logs written before modes existed hold combined runs only.
    if "mode" not in merged.columns:
        merged["mode"] = "combined"
    merged["mode"] = merged["mode"].fillna("combined")

    n_raw = len(merged)
    merged = merged.drop_duplicates(subset=KEY, keep="last")
    merged = merged.sort_values(KEY).reset_index(drop=True)
    merged.to_csv(merged_path, index=False)
    print(f"\nWrote {len(merged)} rows to {merged_path} "
          f"({n_raw - len(merged)} duplicate keys collapsed)")

    if not args.keep_shards:
        for shard in shards:
            shard.unlink()
        print(f"Removed {len(shards)} shard file(s)")

    print("\nRuns per dataset/mode:")
    print(merged.groupby(["dataset", "mode"]).size().to_string())


if __name__ == "__main__":
    main()
