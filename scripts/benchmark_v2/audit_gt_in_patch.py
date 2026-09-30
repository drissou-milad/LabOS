"""
Experiment #5 geometry audit: ALL-GT-in-32x32-patch audit.

READ-ONLY. Inputs: gt_coordinates.csv (from extract_gt_coordinates.py) and manifest.csv (from
build_exp05_training_dataset.py). Does not modify either file, any other LabOS file, any .geff/
.zarr file, or any patch .npy. Does not import torch. Does not train, evaluate, or load any
model. Writes exactly two new output files (a per-candidate CSV and a summary text report) to
paths you choose -- nothing else on disk is touched.

WHY THIS IS INDEPENDENT of manifest.csv's own nearest_gt_y/nearest_gt_x/nearest_gt_distance_px
columns: those columns record the SINGLE closest GT point in the whole frame (Euclidean
argmin over all in-frame GT, computed once by build_exp05_training_dataset.py at label-assignment
time). This script instead re-derives, from gt_coordinates.csv directly, EVERY GT point that
falls inside the specific 32x32 BOX a given candidate's patch actually occupies:

    candidate_y - 16 <= gt_y < candidate_y + 16
    candidate_x - 16 <= gt_x < candidate_x + 16

These are different questions with different answers in general: a candidate's frame-global
nearest GT (Euclidean) can lie just outside its own patch box (e.g. near a box corner), while a
DIFFERENT, farther-by-Euclidean-distance GT point can simultaneously sit inside the box. This
script also reports manifest's own nearest_gt_distance_px alongside the box-derived numbers as a
consistency cross-check (see --sanity-check-tolerance-px), but the box-derived GT-inside-patch
count is the primary output, computed independently of that column.

Usage (run locally, in your existing venv, from the LabOS project root):

    python scripts/benchmark_v2/audit_gt_in_patch.py `
        --manifest results\\exp05_training_dataset\\run02_20samples\\manifest.csv `
        --gt-coordinates results\\exp05_training_dataset\\run02_20samples\\gt_coordinates.csv `
        --out-per-candidate results\\exp05_training_dataset\\run02_20samples\\audit_gt_in_patch_per_candidate.csv `
        --out-summary results\\exp05_training_dataset\\run02_20samples\\audit_gt_in_patch_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATCH_HALF = 16
TEST_SAMPLES_DEFAULT = ["44b6_0b24845f", "44b6_1d530831", "44b6_33b596bf"]
NEG_DISTANCE_BUCKETS = [(10, 12), (12, 16), (16, 20), (20, 32), (32, np.inf)]
POS_DISTANCE_FRACTIONS = [1, 2, 3, 4, 5]
MAX_PAIRS_PER_GROUP_WARN = 2_000_000  # candidates_in_group * gt_in_group, informational only


def load_manifest(path):
    df = pd.read_csv(path)
    required = {"sample_id", "frame_idx", "candidate_y", "candidate_x", "label", "split",
                "nearest_gt_distance_px"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"BLOCKED: {path} is missing required column(s): {sorted(missing)}. "
                          f"Found columns: {list(df.columns)}")
    return df


def load_gt_coordinates(path):
    df = pd.read_csv(path)
    required = {"sample_id", "frame_idx", "gt_y", "gt_x"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"BLOCKED: {path} is missing required column(s): {sorted(missing)}. "
                          f"Found columns: {list(df.columns)}")
    return df


def compute_gt_in_patch(manifest_df, gt_df, half=PATCH_HALF, max_pairs_warn=MAX_PAIRS_PER_GROUP_WARN):
    """For every manifest row, find every GT point (same sample_id + frame_idx) inside that
    candidate's box. Grouped by (sample_id, frame_idx) so each candidate is only ever compared
    against GT points from its own frame -- never across frames or samples, matching the
    labeling policy's own frame-locality."""
    gt_groups = {key: g[["gt_y", "gt_x"]].to_numpy(dtype=np.float64)
                 for key, g in gt_df.groupby(["sample_id", "frame_idx"])}

    n_gt_inside = np.zeros(len(manifest_df), dtype=np.int64)
    nearest_dist_inside = np.full(len(manifest_df), np.nan)
    second_nearest_dist_inside = np.full(len(manifest_df), np.nan)
    nearest_local_y = np.full(len(manifest_df), np.nan)
    nearest_local_x = np.full(len(manifest_df), np.nan)
    n_frames_with_zero_gt_rows = 0

    for key, idx in manifest_df.groupby(["sample_id", "frame_idx"]).groups.items():
        gt_pts = gt_groups.get(key)
        idx = np.asarray(idx)
        if gt_pts is None or len(gt_pts) == 0:
            n_frames_with_zero_gt_rows += len(idx)
            continue

        cand = manifest_df.loc[idx, ["candidate_y", "candidate_x"]].to_numpy(dtype=np.float64)

        if len(cand) * len(gt_pts) > max_pairs_warn:
            print(f"  NOTE: group {key} has {len(cand)} candidates x {len(gt_pts)} GT points "
                  f"({len(cand)*len(gt_pts):,} pairs) -- large but proceeding.")

        dy = np.abs(cand[:, 0][:, None] - gt_pts[:, 0][None, :])
        dx = np.abs(cand[:, 1][:, None] - gt_pts[:, 1][None, :])
        inside = (dy < 2 * half) & (dx < 2 * half) & (
            (cand[:, 0][:, None] - half <= gt_pts[:, 0][None, :]) &
            (gt_pts[:, 0][None, :] < cand[:, 0][:, None] + half) &
            (cand[:, 1][:, None] - half <= gt_pts[:, 1][None, :]) &
            (gt_pts[:, 1][None, :] < cand[:, 1][:, None] + half)
        )
        euclid = np.sqrt((cand[:, 0][:, None] - gt_pts[:, 0][None, :]) ** 2 +
                          (cand[:, 1][:, None] - gt_pts[:, 1][None, :]) ** 2)
        masked = np.where(inside, euclid, np.inf)

        n_inside_row = inside.sum(axis=1)
        n_gt_inside[idx] = n_inside_row

        sorted_d = np.sort(masked, axis=1)
        nearest = sorted_d[:, 0]
        second = sorted_d[:, 1] if sorted_d.shape[1] > 1 else np.full(len(idx), np.inf)
        nearest_dist_inside[idx] = np.where(np.isfinite(nearest), nearest, np.nan)
        second_nearest_dist_inside[idx] = np.where(np.isfinite(second), second, np.nan)

        nearest_gt_idx = np.argmin(masked, axis=1)
        has_any = n_inside_row > 0
        for local_i, has in enumerate(has_any):
            if not has:
                continue
            gi = nearest_gt_idx[local_i]
            gy, gx = gt_pts[gi]
            cy, cx = cand[local_i]
            nearest_local_y[idx[local_i]] = gy - (cy - half)
            nearest_local_x[idx[local_i]] = gx - (cx - half)

    out = manifest_df.copy()
    out["n_gt_inside_patch"] = n_gt_inside
    out["nearest_gt_dist_inside_patch"] = nearest_dist_inside
    out["second_nearest_gt_dist_inside_patch"] = second_nearest_dist_inside
    out["nearest_gt_local_y"] = nearest_local_y
    out["nearest_gt_local_x"] = nearest_local_x
    if n_frames_with_zero_gt_rows:
        print(f"  NOTE: {n_frames_with_zero_gt_rows} manifest rows belong to a (sample_id, "
              f"frame_idx) with ZERO rows in gt_coordinates.csv -- n_gt_inside_patch=0 for "
              f"these by construction (nothing to find), not an error. If this number is large "
              f"relative to manifest.csv's row count, double check gt_coordinates.csv covers "
              f"every frame, not just frames that had a labeled candidate.")
    return out


def sanity_check_against_manifest_nearest(df, tolerance_px=1e-6):
    """Cross-check only: rows where manifest's OWN nearest_gt_distance_px is <= 16 - 0 (i.e.
    definitely small enough it COULD be inside the box on-axis) but n_gt_inside_patch==0 would
    indicate either a coordinate-convention mismatch between manifest.csv and gt_coordinates.csv,
    or a genuine off-axis case (global-nearest just outside the box while box has zero GT) -- the
    latter is expected sometimes since the box isn't a Euclidean disk. Reports counts only, does
    not resolve them."""
    has_manifest_dist = df["nearest_gt_distance_px"].notna()
    plausibly_inside = df["nearest_gt_distance_px"] < PATCH_HALF
    zero_found = df["n_gt_inside_patch"] == 0
    n_flag = int((has_manifest_dist & plausibly_inside & zero_found).sum())
    return n_flag


def pct(x):
    return float(np.mean(x)) * 100.0


def summarize(df, test_samples):
    lines = []
    lines.append("=" * 78)
    lines.append("ALL-GT-IN-32x32-PATCH AUDIT SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Total manifest rows analyzed: {len(df)}")
    lines.append(f"  by label: {df['label'].value_counts().to_dict()}")
    lines.append(f"  by split: {df['split'].value_counts().to_dict()}")

    n_flag = sanity_check_against_manifest_nearest(df)
    lines.append(f"\nConsistency flag: {n_flag} rows where manifest's own "
                 f"nearest_gt_distance_px < {PATCH_HALF}px but this script's independent "
                 f"box search found 0 GT inside the patch. A non-zero but small count is "
                 f"expected (near-corner geometry); a large count would suggest a coordinate "
                 f"convention mismatch between manifest.csv and gt_coordinates.csv and should "
                 f"be investigated before trusting the rest of this report.")

    # --- AUDIT 2: GT-inside-patch histogram, by split x label ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 2 -- GT points inside patch: distribution by split x label")
    lines.append("-" * 78)
    for (split, label), g in df.groupby(["split", "label"]):
        n = len(g)
        buckets = g["n_gt_inside_patch"].clip(upper=3)
        counts = buckets.value_counts().reindex([0, 1, 2, 3], fill_value=0)
        lines.append(f"{split:6s} {label:9s} n={n:7d}  "
                     f"0 GT={counts[0]:7d} ({100*counts[0]/n:5.2f}%)  "
                     f"1 GT={counts[1]:7d} ({100*counts[1]/n:5.2f}%)  "
                     f"2 GT={counts[2]:6d} ({100*counts[2]/n:5.2f}%)  "
                     f"3+ GT={counts[3]:6d} ({100*counts[3]/n:5.2f}%)")

    lines.append(f"\n%% negative patches containing >=1 GT inside the 32x32 patch (overall):")
    neg = df[df["label"] == "negative"]
    if len(neg):
        pct_neg_with_gt = pct(neg["n_gt_inside_patch"] > 0)
        lines.append(f"  {pct_neg_with_gt:.2f}%  (n={len(neg)})")
    for sid in test_samples:
        sub = neg[(neg["sample_id"] == sid)]
        if len(sub):
            lines.append(f"  {sid}: {pct(sub['n_gt_inside_patch'] > 0):.2f}%  (n={len(sub)})")

    # --- AUDIT 3: positive centering (using manifest's own nearest_gt_distance_px, the
    # authoritative labeling-time distance -- NOT the box-derived one, since a positive's true
    # labeling GT could in principle sit just outside its own box in rare edge cases) ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 3 -- Positive centering: nearest_gt_distance_px distribution")
    lines.append("-" * 78)
    pos = df[df["label"] == "positive"]

    def dist_stats(x):
        x = x.dropna()
        if len(x) == 0:
            return None
        return {
            "n": len(x), "mean": x.mean(), "median": x.median(),
            "p25": x.quantile(0.25), "p75": x.quantile(0.75),
            "p90": x.quantile(0.90), "p95": x.quantile(0.95), "max": x.max(),
        }

    s = dist_stats(pos["nearest_gt_distance_px"])
    if s:
        lines.append(f"Overall positives (n={s['n']}): mean={s['mean']:.3f} median={s['median']:.3f} "
                     f"p25={s['p25']:.3f} p75={s['p75']:.3f} p90={s['p90']:.3f} p95={s['p95']:.3f} "
                     f"max={s['max']:.3f}")
    for sid in sorted(df["sample_id"].unique()):
        sub = pos[pos["sample_id"] == sid]
        s = dist_stats(sub["nearest_gt_distance_px"])
        if s:
            tag = "  <-- TEST" if sid in test_samples else ""
            lines.append(f"  {sid}{tag}: mean={s['mean']:.3f} median={s['median']:.3f} "
                         f"p90={s['p90']:.3f} p95={s['p95']:.3f} max={s['max']:.3f} n={s['n']}")

    lines.append("\nFraction of positives within distance thresholds (overall):")
    for r in POS_DISTANCE_FRACTIONS:
        f = pct(pos["nearest_gt_distance_px"] <= r)
        lines.append(f"  <= {r}px: {f:.2f}%")

    # --- AUDIT 4: negative near-GT distribution + buckets ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 4 -- Negative nearest_gt_distance_px distribution + buckets")
    lines.append("-" * 78)
    s = dist_stats(neg["nearest_gt_distance_px"])
    if s:
        lines.append(f"Overall negatives (n={s['n']}): mean={s['mean']:.3f} median={s['median']:.3f} "
                     f"p25={s['p25']:.3f} p75={s['p75']:.3f} p90={s['p90']:.3f} p95={s['p95']:.3f} "
                     f"max={s['max']:.3f} min={neg['nearest_gt_distance_px'].min():.3f}")
    lines.append("Buckets (overall):")
    for lo, hi in NEG_DISTANCE_BUCKETS:
        d = neg["nearest_gt_distance_px"]
        f = pct((d >= lo) & (d < hi))
        hi_label = f"{hi}" if np.isfinite(hi) else "inf"
        lines.append(f"  {lo}-{hi_label}px: {f:.2f}%")
    for sid in test_samples:
        lines.append(f"  -- {sid} --")
        sub = neg[neg["sample_id"] == sid]
        for lo, hi in NEG_DISTANCE_BUCKETS:
            d = sub["nearest_gt_distance_px"]
            f = pct((d >= lo) & (d < hi)) if len(sub) else float("nan")
            hi_label = f"{hi}" if np.isfinite(hi) else "inf"
            lines.append(f"     {lo}-{hi_label}px: {f:.2f}%  (n={len(sub)})")

    # --- AUDIT 5: GT local-position-in-patch, for rows with >=1 GT inside ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 5 -- Local position of nearest in-box GT (candidate is always at (16,16))")
    lines.append("-" * 78)
    has_gt = df["n_gt_inside_patch"] > 0
    for label in ["positive", "negative"]:
        sub = df[(df["label"] == label) & has_gt]
        if len(sub) == 0:
            continue
        ly, lx = sub["nearest_gt_local_y"], sub["nearest_gt_local_x"]
        r = np.hypot(ly - 16, lx - 16)
        lines.append(f"{label}: n(with >=1 GT inside)={len(sub)}  "
                     f"mean local=({ly.mean():.2f},{lx.mean():.2f})  "
                     f"mean radial offset from (16,16)={r.mean():.2f}px  median={np.median(r):.2f}px")

    # --- AUDIT 7/8: label/context conflict categories ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 7/8 -- Label/context conflict categories")
    lines.append("-" * 78)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos:
        lines.append(f"1. positive + multiple GT inside patch: "
                     f"{pct(pos['n_gt_inside_patch'] > 1):.2f}%  ({(pos['n_gt_inside_patch']>1).sum()}/{n_pos})")
    if n_neg:
        n_neg_with_gt = int((neg["n_gt_inside_patch"] > 0).sum())
        lines.append(f"3. negative + >=1 GT inside patch: "
                     f"{pct(neg['n_gt_inside_patch'] > 0):.2f}%  ({n_neg_with_gt}/{n_neg})")
        lines.append(f"   of which negative + multiple (>=2) GT inside patch: "
                     f"{pct(neg['n_gt_inside_patch'] > 1):.2f}%  "
                     f"({(neg['n_gt_inside_patch']>1).sum()}/{n_neg})")

    # --- AUDIT 9: per-sample table ---
    lines.append("\n" + "-" * 78)
    lines.append("AUDIT 9 -- Per-sample summary")
    lines.append("-" * 78)
    lines.append(f"{'sample_id':16s} {'split':6s} {'n_pos':>6s} {'n_neg':>7s} "
                 f"{'pos_mean_dist':>13s} {'neg_mean_dist':>13s} {'%neg_w/GT':>10s}")
    for sid, g in df.groupby("sample_id"):
        split = g["split"].iloc[0]
        gp = g[g["label"] == "positive"]
        gn = g[g["label"] == "negative"]
        pos_mean = gp["nearest_gt_distance_px"].mean() if len(gp) else float("nan")
        neg_mean = gn["nearest_gt_distance_px"].mean() if len(gn) else float("nan")
        neg_gt_pct = pct(gn["n_gt_inside_patch"] > 0) if len(gn) else float("nan")
        tag = " <--TEST" if sid in test_samples else ""
        lines.append(f"{sid:16s} {split:6s} {len(gp):6d} {len(gn):7d} "
                     f"{pos_mean:13.3f} {neg_mean:13.3f} {neg_gt_pct:9.2f}%{tag}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt-coordinates", required=True)
    parser.add_argument("--out-per-candidate", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--labels", default="positive,negative",
                         help="Comma-separated manifest label values to include. Default "
                              "excludes 'ambiguous' (never used for training, per the frozen "
                              "policy) -- pass 'positive,negative,ambiguous' to include it.")
    parser.add_argument("--test-samples", default=",".join(TEST_SAMPLES_DEFAULT))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    gt_path = Path(args.gt_coordinates)
    out_per_candidate = Path(args.out_per_candidate)
    out_summary = Path(args.out_summary)

    for p in [manifest_path, gt_path]:
        if not p.exists():
            raise SystemExit(f"BLOCKED: {p} not found.")
    for p in [out_per_candidate, out_summary]:
        if p.exists():
            raise SystemExit(f"BLOCKED: {p} already exists -- refusing to overwrite. Choose a "
                              f"different output path.")

    print(f"Loading {manifest_path} ...")
    manifest_df = load_manifest(manifest_path)
    labels_wanted = [s.strip() for s in args.labels.split(",")]
    manifest_df = manifest_df[manifest_df["label"].isin(labels_wanted)].reset_index(drop=True)
    print(f"  {len(manifest_df)} rows after filtering to label in {labels_wanted}")

    print(f"Loading {gt_path} ...")
    gt_df = load_gt_coordinates(gt_path)
    print(f"  {len(gt_df)} GT points across {gt_df['sample_id'].nunique()} samples")

    manifest_samples = set(manifest_df["sample_id"].unique())
    gt_samples = set(gt_df["sample_id"].unique())
    if not manifest_samples.issubset(gt_samples):
        missing = manifest_samples - gt_samples
        print(f"  WARNING: {len(missing)} sample_id(s) in manifest.csv have NO rows in "
              f"gt_coordinates.csv: {sorted(missing)}. Their n_gt_inside_patch will be 0 for "
              f"every row (indistinguishable from 'genuinely zero GT nearby' -- verify "
              f"gt_coordinates.csv actually covers these samples).")

    print("\nComputing GT-in-patch for every candidate (grouped by sample_id, frame_idx)...")
    result_df = compute_gt_in_patch(manifest_df, gt_df)

    keep_cols = ["sample_id", "frame_idx", "candidate_y", "candidate_x", "label", "split",
                 "nearest_gt_distance_px", "n_gt_inside_patch", "nearest_gt_dist_inside_patch",
                 "second_nearest_gt_dist_inside_patch", "nearest_gt_local_y", "nearest_gt_local_x"]
    result_df[keep_cols].to_csv(out_per_candidate, index=False)
    print(f"Saved: {out_per_candidate}  ({len(result_df)} rows)")

    test_samples = [s.strip() for s in args.test_samples.split(",") if s.strip()]
    summary_text = summarize(result_df, test_samples)
    out_summary.write_text(summary_text)
    print(f"Saved: {out_summary}")
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
