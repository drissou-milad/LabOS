from pathlib import Path
import numpy as np
import pandas as pd
import zarr

DATASET_ROOT = Path(
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development"
)

PREDICTIONS = Path(
    r"results\exp05_training_dataset\run02_20samples"
    r"\run13_test_eval\test_predictions_run13.csv"
)

DISTANCES = Path(
    r"results\exp05_audit\audit_first20\candidate_distances.csv"
)

OUT_DIR = Path(
    r"results\exp05_training_dataset\run02_20samples"
    r"run13_temporal_audit_expanded"
)

FRAME_RADIUS = 3
PATCH_SIZE = 32

# Four cases already investigated in the completed Run13 audits.
EXCLUDED = {
    ("44b6_33b596bf", 43, 109, 89),
    ("44b6_1d530831", 35, 56, 211),
    ("44b6_1d530831", 92, 139, 143),
    ("44b6_1d530831", 55, 168, 73),
}


def load_gt(sample_id):
    path = DATASET_ROOT / "train" / f"{sample_id}.geff"
    graph = zarr.open_group(path, mode="r")
    props = graph["nodes"]["props"]

    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    by_frame = {}

    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append(
            (float(y), float(x))
        )

    return {
        frame: np.asarray(points, dtype=float).reshape(-1, 2)
        for frame, points in by_frame.items()
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PREDICTIONS)
    dist = pd.read_csv(DISTANCES)

    negatives = (
        pred[pred["label"].astype(float) == 0.0]
        .sort_values("probability", ascending=False)
        .head(30)
        .copy()
    )

    selected = []

    for _, row in negatives.iterrows():
        filename = str(row["filename"])

        stem = Path(filename).stem
        parts = stem.rsplit("_", 3)

        if len(parts) != 4:
            raise ValueError(
                f"Cannot parse candidate filename: {filename}"
            )

        sample_id = parts[0]
        frame_idx = int(parts[1])
        candidate_y = int(parts[2])
        candidate_x = int(parts[3])

        key = (
            sample_id,
            frame_idx,
            candidate_y,
            candidate_x,
        )

        if key in EXCLUDED:
            continue

        match = dist[
            (dist["sample"] == sample_id)
            & (dist["frame"].astype(int) == frame_idx)
            & (dist["y"].astype(int) == candidate_y)
            & (dist["x"].astype(int) == candidate_x)
        ]

        if len(match) == 0:
            raise ValueError(
                f"No distance record found for {key}"
            )

        distance = float(
            match.iloc[0]["nearest_gt_distance_px"]
        )

        selected.append(
            {
                "sample_id": sample_id,
                "center_frame": frame_idx,
                "candidate_y": candidate_y,
                "candidate_x": candidate_x,
                "probability": float(row["probability"]),
                "nearest_gt_distance_px": distance,
                "filename": filename,
            }
        )

    # Stratified inspection set:
    # near, intermediate, and far from annotated GT.
    selected_df = pd.DataFrame(selected)

    near = selected_df[
        selected_df["nearest_gt_distance_px"] < 60
    ].sort_values("nearest_gt_distance_px").head(4)

    intermediate = selected_df[
        (selected_df["nearest_gt_distance_px"] >= 60)
        & (selected_df["nearest_gt_distance_px"] < 100)
    ].sort_values("nearest_gt_distance_px").head(4)

    far = selected_df[
        selected_df["nearest_gt_distance_px"] >= 100
    ].sort_values(
        "probability", ascending=False
    ).head(4)

    cases = pd.concat(
        [near, intermediate, far],
        ignore_index=True,
    )

    all_gt = {}
    rows = []

    for sample_id in cases["sample_id"].unique():
        all_gt[sample_id] = load_gt(sample_id)

    for _, case in cases.iterrows():
        sample_id = case["sample_id"]
        center = int(case["center_frame"])
        cy = int(case["candidate_y"])
        cx = int(case["candidate_x"])
        probability = float(case["probability"])

        gt_by_frame = all_gt[sample_id]

        # Infer available frame range from GEFF annotations.
        available_frames = sorted(gt_by_frame.keys())

        if not available_frames:
            raise ValueError(
                f"No GT frames found for {sample_id}"
            )

        min_frame = min(available_frames)
        max_frame = max(available_frames)

        frames = range(
            max(min_frame, center - FRAME_RADIUS),
            min(max_frame, center + FRAME_RADIUS) + 1,
        )

        half = PATCH_SIZE // 2

        for frame_idx in frames:
            gt_pts = gt_by_frame.get(
                frame_idx,
                np.empty((0, 2), dtype=float),
            )

            if len(gt_pts):
                distances = np.sqrt(
                    (gt_pts[:, 0] - cy) ** 2
                    + (gt_pts[:, 1] - cx) ** 2
                )

                nearest_idx = int(np.argmin(distances))
                nearest_distance = float(
                    distances[nearest_idx]
                )
                nearest_y = float(
                    gt_pts[nearest_idx, 0]
                )
                nearest_x = float(
                    gt_pts[nearest_idx, 1]
                )

                y1, y2 = cy - half, cy + half
                x1, x2 = cx - half, cx + half

                inside = (
                    (gt_pts[:, 0] >= y1)
                    & (gt_pts[:, 0] < y2)
                    & (gt_pts[:, 1] >= x1)
                    & (gt_pts[:, 1] < x2)
                )

                n_inside = int(inside.sum())

                inside_distances = distances[inside]

                nearest_inside = (
                    float(inside_distances.min())
                    if len(inside_distances)
                    else np.nan
                )

            else:
                nearest_distance = np.nan
                nearest_y = np.nan
                nearest_x = np.nan
                n_inside = 0
                nearest_inside = np.nan

            rows.append(
                {
                    "sample_id": sample_id,
                    "case_center_frame": center,
                    "frame_idx": int(frame_idx),
                    "candidate_y_fixed": cy,
                    "candidate_x_fixed": cx,
                    "probability": probability,
                    "center_frame_nearest_gt_distance_px":
                        float(case["nearest_gt_distance_px"]),
                    "n_gt_in_frame": int(len(gt_pts)),
                    "nearest_gt_distance_to_fixed_candidate_px":
                        nearest_distance,
                    "nearest_gt_y": nearest_y,
                    "nearest_gt_x": nearest_x,
                    "n_gt_inside_fixed_32px_patch": n_inside,
                    "nearest_gt_inside_patch_distance_px":
                        nearest_inside,
                }
            )

    context_df = pd.DataFrame(rows)

    selected_path = OUT_DIR / "selected_cases.csv"
    context_path = OUT_DIR / "expanded_temporal_gt_context.csv"

    cases.to_csv(selected_path, index=False)
    context_df.to_csv(context_path, index=False)

    print("Expanded Run13 temporal GT audit")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Selected cases:   {len(cases)}")
    print(f"Context rows:     {len(context_df)}")
    print()
    print("Selected cases:")
    print(
        cases[
            [
                "sample_id",
                "center_frame",
                "candidate_y",
                "candidate_x",
                "probability",
                "nearest_gt_distance_px",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Saved: {selected_path}")
    print(f"Saved: {context_path}")
    print()
    print("READ-ONLY AUDIT: no model, checkpoint, dataset,")
    print("benchmark prediction, or existing audit file modified.")


if __name__ == "__main__":
    main()
