from pathlib import Path
import argparse


def convert_results(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for src_file in src_root.rglob("state_estimate.txt"):
        rel_path = src_file.relative_to(src_root)
        dst_file = (dst_root / rel_path).with_name("poses.txt")
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with open(src_file, "r") as fin, open(dst_file, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                timestamp_ns = int(round(float(fields[0]) * 1e9))
                qx, qy, qz, qw = fields[1:5]
                px, py, pz = fields[5:8]
                fout.write(
                    f"{timestamp_ns} "
                    f"{px} {py} {pz} "
                    f"{qx} {qy} {qz} {qw}\n"
                )

        print(f"Converted: {src_file} -> {dst_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OpenVINS results")
    parser.add_argument(
        "--src_root",
        required=True,
        help="Source root directory",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        help="Destination root directory",
    )
    args = parser.parse_args()
    convert_results(args.src_root, args.dst_root)
