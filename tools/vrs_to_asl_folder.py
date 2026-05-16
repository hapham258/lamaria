import argparse
import csv
import os
import shutil
from pathlib import Path

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from tqdm import tqdm

from lamaria import logger
from lamaria.utils.aria import extract_images_from_vrs
from lamaria.utils.constants import (
    LEFT_CAMERA_STREAM_ID,
    RIGHT_CAMERA_STREAM_ID,
    RIGHT_IMU_STREAM_ID,
)
from lamaria.utils.timestamps import (
    get_matched_timestamps,
)


def remove_images_when_slam_drops(
    image_folder: Path,
    left_timestamps: list[int],
    right_timestamps: list[int],
    matched_timestamps: list[tuple[int, int]],
    left_subfolder_name="cam0/data",
    right_subfolder_name="cam1/data",
):
    left_image_folder = os.path.join(image_folder, left_subfolder_name)
    right_image_folder = os.path.join(image_folder, right_subfolder_name)

    assert os.path.exists(left_image_folder) and os.path.exists(
        right_image_folder
    )

    original_left_images = sorted(os.listdir(left_image_folder))
    original_right_images = sorted(os.listdir(right_image_folder))

    assert len(original_left_images) == len(left_timestamps)
    assert len(original_right_images) == len(right_timestamps)

    left_camera_mapping = {
        ts: img for ts, img in zip(left_timestamps, original_left_images)
    }
    right_camera_mapping = {
        ts: img for ts, img in zip(right_timestamps, original_right_images)
    }

    matched_left_ts = [left_ts for left_ts, _ in matched_timestamps]
    matched_right_ts = [right_ts for _, right_ts in matched_timestamps]

    if len(left_timestamps) != len(matched_timestamps):
        for ts, img in left_camera_mapping.items():
            if ts not in matched_left_ts:
                print(f"Removing {img}")
                os.remove(os.path.join(left_image_folder, img))
    elif len(right_timestamps) != len(matched_timestamps):
        for ts, img in right_camera_mapping.items():
            if ts not in matched_right_ts:
                print(f"Removing {img}")
                os.remove(os.path.join(right_image_folder, img))
    else:
        raise ValueError("No images to remove")

    return matched_left_ts


def rename_images_in_folder(
    aria_folder: Path,
    image_timestamps,
    left_subfolder_name="cam0/data",
    right_subfolder_name="cam1/data",
    image_extension=".jpg",
) -> list[int]:
    for subfolder in [left_subfolder_name, right_subfolder_name]:
        subfolder_path = aria_folder / subfolder

        if not subfolder_path.exists() and not subfolder_path.is_dir():
            raise ValueError(
                f"{subfolder_path} does not exist or is not a directory"
            )

        original_images = sorted(
            [
                f
                for f in os.listdir(subfolder_path)
                if f.endswith(image_extension)
            ]
        )
        if len(original_images) == 0:
            raise ValueError(f"No images found in {subfolder_path}")

        if len(original_images) != len(image_timestamps):
            raise ValueError(
                f"Number of images {len(original_images)} \
                    does not match number of timestamps \
                    {len(image_timestamps)} in {subfolder_path}"
            )

        for ts, img in zip(image_timestamps, original_images):
            old_image_path = subfolder_path / img
            new_image_path = subfolder_path / f"{ts}{image_extension}"
            os.rename(old_image_path, new_image_path)

    return image_timestamps


def write_image_timestamps_to_txt(image_timestamps: list, txt_file: Path):
    with open(txt_file, "w") as f:
        for timestamp in image_timestamps:
            f.write(f"{timestamp}\n")


def write_image_csv(image_timestamps, cam_folder):
    images = os.listdir(os.path.join(cam_folder, "data"))
    assert len(images) > 0

    images = sorted(images, key=lambda img: int(img.split(".")[0]))

    assert len(images) == len(image_timestamps)
    for ts, img in zip(image_timestamps, images):
        assert int(img.split(".")[0]) == ts, f"{img} != {ts}"

    data_csv = os.path.join(cam_folder, "data.csv")
    if os.path.exists(data_csv):
        os.remove(data_csv)

    with open(data_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "filename"])
        for timestamp, image in zip(image_timestamps, images):
            row = [timestamp, image]
            writer.writerow(row)


def write_imu_data_to_csv(vrs_provider, csv_file):
    imu_timestamps = vrs_provider.get_timestamps_ns(
        StreamId(RIGHT_IMU_STREAM_ID), TimeDomain.DEVICE_TIME
    )

    last_timestamp = None
    if os.path.exists(csv_file):
        with open(csv_file) as f:
            last_row = None
            for _ in csv.reader(f):
                pass
            if last_row is not None:
                last_timestamp = int(last_row[0])

    if last_timestamp is not None:
        imu_timestamps = [ts for ts in imu_timestamps if ts > last_timestamp]

    if not imu_timestamps:
        logger.info(f"No new IMU data to write to {csv_file}")
        return

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        for timestamp in tqdm(imu_timestamps, desc="Appending IMU data to CSV"):
            imu_data = vrs_provider.get_imu_data_by_time_ns(
                StreamId(RIGHT_IMU_STREAM_ID),
                timestamp,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST,
            )
            if imu_data.accel_valid and imu_data.gyro_valid:
                accel = imu_data.accel_msec2
                gyro = imu_data.gyro_radsec

                row = [
                    timestamp,
                    gyro[0],
                    gyro[1],
                    gyro[2],
                    accel[0],
                    accel[1],
                    accel[2],
                ]
                writer.writerow(row)


def write_exposure_csv(
    vrs_file: Path,
    stream_label: str,
    csv_path: Path,
):
    provider = data_provider.create_vrs_data_provider(vrs_file.as_posix())
    stream_id = provider.get_stream_id_from_label(stream_label)
    num_frames = provider.get_num_data(stream_id)
    with open(csv_path, "w") as f:
        f.write("#timestamp [ns],exposure time[ns]\n")
        for i in range(num_frames):
            _, metadata = provider.get_image_data_by_index(stream_id, i)
            timestamp_ns = metadata.capture_timestamp_ns
            exposure_ns = int(metadata.exposure_duration * 1e9)
            f.write(f"{timestamp_ns},{exposure_ns}\n")


def form_aria_asl_folder(
    vrs_file: Path, output_asl_folder: Path, has_slam_drops=False
):
    if output_asl_folder.exists():
        print(f"{output_asl_folder=} already exists. Skipping.")
        return

    aria_folder = output_asl_folder / "mav0"
    aria_folder.mkdir(parents=True, exist_ok=True)

    dataset_name = vrs_file.stem
    vrs_provider = data_provider.create_vrs_data_provider(vrs_file.as_posix())

    # Get all image timestamps (in ns)
    image_timestamps = vrs_provider.get_timestamps_ns(
        StreamId(LEFT_CAMERA_STREAM_ID), TimeDomain.DEVICE_TIME
    )
    assert len(image_timestamps) > 0, "No timestamps found"

    right_image_timestamps = None
    matched_timestamps = None
    if has_slam_drops:
        right_image_timestamps = vrs_provider.get_timestamps_ns(
            StreamId(RIGHT_CAMERA_STREAM_ID), TimeDomain.DEVICE_TIME
        )
        assert len(right_image_timestamps) > 0, (
            "No right camera image timestamps found"
        )
        assert len(right_image_timestamps) != len(image_timestamps), (
            "Left and right camera image timestamps are the same"
        )
        matched_timestamps = get_matched_timestamps(
            left_timestamps=image_timestamps,
            right_timestamps=right_image_timestamps,
            max_diff=1e6,  # 1 ms in nanoseconds
        )

        assert len(matched_timestamps) > 0, "No matched timestamps found"

    extract_images_from_vrs(
        vrs_file=vrs_file,
        image_folder=aria_folder,
        left_subfolder_name="cam0/data",
        right_subfolder_name="cam1/data",
    )

    if has_slam_drops:
        assert (
            right_image_timestamps is not None
            and matched_timestamps is not None
        )
        image_timestamps = remove_images_when_slam_drops(
            aria_folder,
            image_timestamps,
            right_image_timestamps,
            matched_timestamps,
        )

    image_timestamps = rename_images_in_folder(
        aria_folder,
        image_timestamps,
    )

    imu_folder = output_asl_folder / "mav0" / "imu0"
    imu_folder.mkdir(parents=True, exist_ok=True)
    imu_csv = imu_folder / "data.csv"

    write_imu_data_to_csv(
        vrs_provider,
        imu_csv,
    )

    write_image_timestamps_to_txt(
        image_timestamps,
        aria_folder / f"{dataset_name}.txt",
    )
    write_image_csv(image_timestamps, aria_folder / "cam0")
    write_image_csv(image_timestamps, aria_folder / "cam1")

    write_exposure_csv(vrs_file, "camera-slam-left", aria_folder / "cam0" / "exposure.csv")
    write_exposure_csv(vrs_file, "camera-slam-right", aria_folder / "cam1" / "exposure.csv")


def convert_imu_csv_to_dso(asl_folder: Path):
    imu_csv = asl_folder / "mav0" / "imu0" / "data.csv"
    dso_dir = asl_folder / "dso"
    dso_dir.mkdir(exist_ok=True)
    imu_txt = dso_dir / "imu_orig.txt"
    with open(imu_csv, "r") as f_in, open(imu_txt, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            timestamp = parts[0]
            gyro = parts[1:4]
            accel = parts[4:7]
            out_line = " ".join([timestamp] + gyro + accel)
            f_out.write(out_line + "\n")
    print(f"Saved: {imu_txt}")


def make_dso_times(asl_folder: Path, cam_name="cam0"):
    data_path = asl_folder / "mav0" / cam_name / "data.csv"
    exposure_path = asl_folder / "mav0" / cam_name / "exposure.csv"
    dso_cam_dir = asl_folder / "dso" / cam_name
    dso_cam_dir.mkdir(parents=True, exist_ok=True)
    times_txt = dso_cam_dir / "times.txt"

    with open(data_path, "r") as f_csv:
        data_lines = [
            line.strip()
            for line in f_csv
            if line.strip() and not line.startswith("#")
        ]
    with open(exposure_path, "r") as f_exp:
        exposure_lines = [
            line.strip()
            for line in f_exp
            if line.strip() and not line.startswith("#")
        ]
    assert len(data_lines) == len(exposure_lines)

    with open(times_txt, "w") as f_out:
        f_out.write("# filename timestamp[s] exposure[ms]\n")
        for data_line, exp_line in zip(data_lines, exposure_lines):
            timestamp_ns, filename = data_line.split(",")
            _, exposure_ns = exp_line.split(",")
            stem = Path(filename).stem
            timestamp_s = int(timestamp_ns) / 1e9
            exposure_ms = int(exposure_ns) / 1e6
            f_out.write(
                f"{stem} "
                f"{timestamp_s:.9f} "
                f"{exposure_ms:.6f}\n"
            )
    print(f"Saved: {times_txt}")


def create_dso_structure(asl_folder: Path):
    """
    Create a TUM-VI/DSO-style folder structure with symlinks.
    """

    mav0_dir = asl_folder / "mav0"    
    cam0_src = mav0_dir / "cam0" / "data"
    cam1_src = mav0_dir / "cam1" / "data"
    if not cam0_src.exists():
        raise FileNotFoundError(f"Missing: {cam0_src}")
    if not cam1_src.exists():
        raise FileNotFoundError(f"Missing: {cam1_src}")

    # Create directories
    dso_dir = asl_folder / "dso"
    (dso_dir / "cam0").mkdir(parents=True, exist_ok=True)
    (dso_dir / "cam1").mkdir(parents=True, exist_ok=True)

    # Symlink targets (relative paths like TUM-VI)
    cam0_link = dso_dir / "cam0" / "images"
    cam1_link = dso_dir / "cam1" / "images"
    cam0_target = Path("../../mav0/cam0/data")
    cam1_target = Path("../../mav0/cam1/data")

    # Remove existing links/files if needed
    for link_path in [cam0_link, cam1_link]:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()

    # Create symlinks
    os.symlink(cam0_target, cam0_link)
    os.symlink(cam1_target, cam1_link)
    print(f"Created: {cam0_link} -> {cam0_target}")
    print(f"Created: {cam1_link} -> {cam1_target}")
    
    # Convert IMU data
    convert_imu_csv_to_dso(asl_folder)

    # Convert to times.txt files
    make_dso_times(args.output_asl_folder, "cam0")
    make_dso_times(args.output_asl_folder, "cam1")


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--vrs_file",
        type=Path,
        required=True,
        help="Path to input VRS file",
    )
    args.add_argument(
        "--output_asl_folder",
        type=Path,
        required=True,
        help="Path to output ASL folder",
    )
    args.add_argument(
        "--has_slam_drops",
        action="store_true",
        help="Whether the VRS file has dropped SLAM frames",
    )

    args = args.parse_args()

    form_aria_asl_folder(
        args.vrs_file,
        args.output_asl_folder,
        has_slam_drops=args.has_slam_drops,
    )

    create_dso_structure(args.output_asl_folder)
