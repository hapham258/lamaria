from pathlib import Path
import argparse
import os
import matplotlib.pyplot as plt
from evo.tools import plot
from evo.core import metrics, sync
from lamaria.structs.trajectory import Trajectory
from lamaria.eval.evo_evaluation import convert_trajectory_to_evo_posetrajectory


def plot_trajectory(
    est_traj,
    gt_traj,
    output_file,
    title="",
):
    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    ax.set_title(title)
    plot.traj(ax, plot_mode, gt_traj, "--", "gray", "Ground Truth")
    plot.traj(ax, plot_mode, est_traj, "-", "blue", "Estimated")
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
        results_root.rglob("poses.txt"),
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
        est_traj = Trajectory.load_from_file(est_file, invert_poses=False)
        gt_traj = Trajectory.load_from_file(gt_file, invert_poses=False)
        est_pose_traj = convert_trajectory_to_evo_posetrajectory(est_traj)
        gt_pose_traj = convert_trajectory_to_evo_posetrajectory(gt_traj)
        gt_pose_traj_sync, est_pose_traj_sync = sync.associate_trajectories(
            gt_pose_traj,
            est_pose_traj,
            max_diff=5_000_000,
        )
        est_pose_traj_sync.align(gt_pose_traj_sync, correct_scale=True)
        pose_relation = metrics.PoseRelation.translation_part
        trajectories = (gt_pose_traj_sync, est_pose_traj_sync)
        ate_metric = metrics.APE(pose_relation)
        ate_metric.process_data(trajectories)
        ate_stats = ate_metric.get_all_statistics()
        ate_rmse = ate_stats["rmse"]

        #
        plot_file = est_file.with_suffix(".pdf")
        plot_trajectory(
            est_pose_traj_sync,
            gt_pose_traj_sync,
            plot_file,
            title=f"{sequence} [{timestamp}], ATE={ate_rmse:.3f}m",
        )
        print(f"Seq: {sequence}, stamp: {timestamp}, ATE: {ate_rmse:.3f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        required=True,
        help="Root containing poses.txt files",
    )
    parser.add_argument(
        "--datasets_root",
        default=os.environ.get("SLAM_DATASETS_PATH"),
        help="SLAM_DATASETS_PATH root",
    )
    args = parser.parse_args()
    main(args.results_root, args.datasets_root)
