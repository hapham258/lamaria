from pathlib import Path
import subprocess
import argparse
import os
import matplotlib.pyplot as plt
from evo.core import sync
from evo.tools import file_interface, plot


def plot_trajectory(
    pred_file,
    gt_file,
    output_file,
    title="",
    align=True,
    correct_scale=True,
):
    #
    pred_traj = file_interface.read_tum_trajectory_file(str(pred_file))
    gt_traj = file_interface.read_tum_trajectory_file(str(gt_file))
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
    if align:
        pred_traj.align(gt_traj, correct_scale=correct_scale)

    #
    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, "-", "blue", "Predicted")
    plot_collection.add_figure("trajectory", fig)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_collection.export(
        str(output_file),
        confirm_overwrite=False,
    )
    plt.close(fig)


def main(results_root, datasets_root):
    results_root = Path(results_root)
    datasets_root = Path(datasets_root)
    est_files = sorted(
        results_root.rglob("resultScaled.txt"),
        key=lambda p: (p.parents[1].name, p.parent.name),
    )
    for est_file in est_files:
        #
        sequence = est_file.parents[1].name
        timestamp = est_file.parent.name
        gt_file = (
            datasets_root
            / "lamaria"
            / "training"
            / sequence
            / "ground_truth"
            / "pGT"
            / f"{sequence}.txt"
        )
        if not gt_file.exists():
            print(f"[WARNING] GT not found: {gt_file}")
            continue

        #
        cmd = [
            "python",
            "-m",
            "evaluate_wrt_mps",
            "--estimate",
            str(est_file),
            "--gt_estimate",
            str(gt_file),
        ]
        print(f"Evaluate {sequence} ({timestamp}) ...")
        subprocess.run(cmd, check=True)

        #
        plot_file = est_file.with_suffix(".pdf")
        plot_trajectory(
            pred_file=est_file,
            gt_file=gt_file,
            output_file=plot_file,
            title=f"{sequence} ({timestamp})",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        required=True,
        help="Root containing resultScaled.txt files",
    )
    parser.add_argument(
        "--datasets_root",
        default=os.environ.get("SLAM_DATASETS_PATH"),
        help="SLAM_DATASETS_PATH root",
    )
    args = parser.parse_args()
    main(args.results_root, args.datasets_root)
