from pathlib import Path
import argparse


def convert_results(src_root, dst_root):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for src_file in src_root.rglob("resultScaled.txt"):
        rel_path = src_file.relative_to(src_root)
        dst_file = dst_root / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        with open(src_file, "r") as fin, open(dst_file, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                timestamp_ns = int(round(float(fields[0]) * 1e9))
                fout.write(f"{timestamp_ns} {' '.join(fields[1:])}\n")
        print(f"Converted: {src_file} -> {dst_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DSO results")
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
