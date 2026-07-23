#!/usr/bin/env python3
"""Exhaustive equal/geometric snapshot averaging selected on Probex only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def weights(count: int, q: float) -> np.ndarray:
    powers = np.arange(count - 1, -1, -1, dtype=np.float64)
    w = np.power(q, powers)
    return w / w.sum()


def gram_rmse(gram: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(max(float(w @ gram @ w), 0.0)))


def error_gram(paths: list[Path], truth: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    gram = np.zeros((len(paths), len(paths)), dtype=np.float64)
    for lo in range(0, len(truth), chunk):
        hi = min(lo + chunk, len(truth))
        y = np.asarray(truth[lo:hi], dtype=np.float64)
        errors = np.stack(
            [np.asarray(array[lo:hi], dtype=np.float64) - y for array in arrays]
        )
        gram += errors @ errors.T
    return gram / len(truth)


def combine(paths: list[Path], w: np.ndarray) -> np.ndarray:
    first = np.load(paths[0], mmap_mode="r")
    out = np.zeros(len(first), dtype=np.float64)
    for weight, path in zip(w, paths):
        out += weight * np.load(path, mmap_mode="r")
    return out.astype(np.float32)


def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        truth = truth[mask]
    error = pred.astype(np.float64) - truth.astype(np.float64)
    return float(np.sqrt(np.mean(error * error)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--preds-dir", type=Path, default=Path("preds_new"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    epochs = list(range(args.first, args.last + 1))
    probex_paths = [
        args.preds_dir / f"{args.model}_ep{epoch:02}.probex.npy" for epoch in epochs
    ]
    qual_paths = [
        args.preds_dir / f"{args.model}_ep{epoch:02}.qual.npy" for epoch in epochs
    ]
    for path in probex_paths + qual_paths:
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    probex_truth = np.load(args.data_dir / "probex" / "ratings.npy", mmap_mode="r")
    qual_truth = np.load(args.data_dir / "qual" / "ratings.npy", mmap_mode="r")
    qual_is_test = np.load(args.data_dir / "qual" / "is_test.npy", mmap_mode="r")
    quiz_mask = qual_is_test == 0

    gram = error_gram(probex_paths, probex_truth)
    rows: list[dict[str, float | int | str]] = []

    for start in epochs:
        for end in range(start, args.last + 1):
            lo = start - args.first
            hi = end - args.first + 1
            subgram = gram[lo:hi, lo:hi]
            count = end - start + 1

            equal_score = gram_rmse(subgram, weights(count, 1.0))
            rows.append(
                {
                    "method": "equal",
                    "start": start,
                    "end": end,
                    "q": 1.0,
                    "probex_rmse": equal_score,
                }
            )

            def objective(q: float) -> float:
                return gram_rmse(subgram, weights(count, q))

            result = minimize_scalar(
                objective, bounds=(0.0, 1.0), method="bounded",
                options={"xatol": 1e-9},
            )
            candidates = [
                (float(result.x), float(result.fun)),
                (0.0, objective(0.0)),
                (1.0, equal_score),
            ]
            q, score = min(candidates, key=lambda item: item[1])
            rows.append(
                {
                    "method": "geometric",
                    "start": start,
                    "end": end,
                    "q": q,
                    "probex_rmse": score,
                }
            )

    best_equal = min(
        (row for row in rows if row["method"] == "equal"),
        key=lambda row: float(row["probex_rmse"]),
    )
    best_geo = min(
        (row for row in rows if row["method"] == "geometric"),
        key=lambda row: float(row["probex_rmse"]),
    )
    standalone = [
        (epoch, gram_rmse(gram[i:i + 1, i:i + 1], np.ones(1)))
        for i, epoch in enumerate(epochs)
    ]
    best_epoch, best_epoch_probex = min(standalone, key=lambda item: item[1])

    def materialize(row: dict[str, float | int | str], suffix: str) -> tuple[float, Path, Path]:
        start = int(row["start"])
        end = int(row["end"])
        q = float(row["q"])
        lo = start - args.first
        hi = end - args.first + 1
        w = weights(end - start + 1, q)
        probex_pred = combine(probex_paths[lo:hi], w)
        qual_pred = combine(qual_paths[lo:hi], w)
        probex_out = args.preds_dir / f"{args.model}__{suffix}.probex.npy"
        qual_out = args.preds_dir / f"{args.model}__{suffix}.qual.npy"
        np.save(probex_out, probex_pred)
        np.save(qual_out, qual_pred)
        return rmse(qual_pred, qual_truth, quiz_mask), probex_out, qual_out

    equal_quiz, _, _ = materialize(best_equal, "equal")
    geo_quiz, _, _ = materialize(best_geo, "ema")
    best_epoch_quiz = rmse(
        np.load(args.preds_dir / f"{args.model}_ep{best_epoch:02}.qual.npy", mmap_mode="r"),
        qual_truth,
        quiz_mask,
    )
    final_probex = standalone[-1][1]
    final_quiz = rmse(
        np.load(args.preds_dir / f"{args.model}_ep{args.last:02}.qual.npy", mmap_mode="r"),
        qual_truth,
        quiz_mask,
    )

    csv_path = args.preds_dir / f"{args.model}_epoch_average_search.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, (ax_epoch, ax_search) = plt.subplots(1, 2, figsize=(14, 6))
    ax_epoch.plot(
        [epoch for epoch, _ in standalone],
        [score for _, score in standalone],
        marker="o", color="#2563eb",
    )
    ax_epoch.set(title="Individual epochs", xlabel="Epoch", ylabel="Probex RMSE")
    ax_epoch.grid(True, alpha=0.25)

    equal_by_end = []
    geo_by_end = []
    for end in epochs:
        equal_by_end.append(
            min(
                float(row["probex_rmse"]) for row in rows
                if row["method"] == "equal" and int(row["end"]) == end
            )
        )
        geo_by_end.append(
            min(
                float(row["probex_rmse"]) for row in rows
                if row["method"] == "geometric" and int(row["end"]) == end
            )
        )
    ax_search.plot(epochs, equal_by_end, label="Best equal interval ending at N")
    ax_search.plot(epochs, geo_by_end, label="Best geometric interval ending at N")
    ax_search.set(title="Probex-only interval search", xlabel="Ending epoch N", ylabel="Probex RMSE")
    ax_search.grid(True, alpha=0.25)
    ax_search.legend()
    fig.suptitle(f"{args.model}: snapshot averaging (Test excluded)")
    fig.tight_layout()
    png_path = args.preds_dir / f"{args.model}_epoch_average_search.png"
    fig.savefig(png_path, dpi=180)

    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")
    print(
        f"Final epoch {args.last}: Probex {final_probex:.9f}, Quiz {final_quiz:.9f}"
    )
    print(
        f"Best epoch {best_epoch}: Probex {best_epoch_probex:.9f}, "
        f"Quiz {best_epoch_quiz:.9f}"
    )
    print(
        f"Best equal: {best_equal['start']}–{best_equal['end']}, "
        f"Probex {float(best_equal['probex_rmse']):.9f}, Quiz {equal_quiz:.9f}"
    )
    print(
        f"Best geometric: {best_geo['start']}–{best_geo['end']}, "
        f"q={float(best_geo['q']):.9f}, "
        f"Probex {float(best_geo['probex_rmse']):.9f}, Quiz {geo_quiz:.9f}"
    )


if __name__ == "__main__":
    main()
