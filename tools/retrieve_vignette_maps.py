import argparse
import cv2
import numpy as np

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mask_path",
        help="Path to devignetting .bin file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height",
    )
    parser.add_argument(
        "--output",
        default="vignette.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    #
    mask = np.fromfile(
        args.mask_path,
        dtype=np.float32,
    )
    mask = mask.reshape(
        args.height,
        args.width,
        3,
    )
    mask_gray = mask.mean(axis=2)
    valid = mask_gray > 0
    inv_mask = np.zeros_like(mask_gray)
    inv_mask[valid] = 1.0 / mask_gray[valid]
    mask_vis = inv_mask / inv_mask.max()
    mask_vis = (mask_vis * 255).astype(np.uint8)
    cv2.imwrite(args.output, mask_vis)
    print(f"Saved: {args.output}")
    print(f"shape={mask.shape} " f"min={mask.min()} " f"max={mask.max()}")
    binary_mask = (valid.astype(np.uint8)) * 255
    inv_valid = (~valid).astype(np.uint8) * 255
    cv2.imwrite("invalid_mask.png", inv_valid)
    print("Saved: invalid_mask.png")
