#!/usr/bin/env python3
"""
Script to resize images for face swap project.
- Source images: resized to 112x112
- Target images: resized to 256x256
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def resize_images(
    src_dir, src_size=112, target_dir=None, target_size=256, output_dir=None
):
    """
    Resize images in source and target directories.

    Args:
        src_dir: Directory containing source images
        src_size: Target size for source images (default: 112)
        target_dir: Directory containing target images (default: None)
        target_size: Target size for target images (default: 256)
        output_dir: Output directory to save resized images (default: None, save in place)
    """

    # Process source images
    if os.path.exists(src_dir):
        print(f"Processing source images from {src_dir}...")
        process_folder(src_dir, src_size, output_dir)
    else:
        print(f"Source directory {src_dir} not found!")

    # Process target images
    if target_dir and os.path.exists(target_dir):
        print(f"Processing target images from {target_dir}...")
        process_folder(target_dir, target_size, output_dir)
    elif target_dir:
        print(f"Target directory {target_dir} not found!")


def process_folder(folder_path, size, output_dir=None):
    """
    Resize all images in a folder to specified size.

    Args:
        folder_path: Path to folder containing images
        size: Target size (will create size x size square)
        output_dir: Output directory (if None, overwrite original)
    """

    supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_count = 0

    for filename in os.listdir(folder_path):
        file_ext = Path(filename).suffix.lower()

        if file_ext not in supported_formats:
            continue

        file_path = os.path.join(folder_path, filename)

        try:
            # Open and resize image
            img = Image.open(file_path)
            resized_img = img.resize((size, size), Image.Resampling.LANCZOS)

            # Determine output path
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
            else:
                output_path = file_path

            # Save resized image
            resized_img.save(output_path, quality=95)
            print(f"  ✓ Resized: {filename} -> {size}x{size}")
            image_count += 1

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    print(f"Processed {image_count} images in {folder_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images for face swap project")

    parser.add_argument(
        "--src-dir",
        type=str,
        default="src/test/src_images",
        help="Source images directory (default: src/test/src_images)",
    )
    parser.add_argument(
        "--src-size", type=int, default=112, help="Source image size (default: 112)"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="src/test/target_images",
        help="Target images directory (default: src/test/target_images)",
    )
    parser.add_argument(
        "--target-size", type=int, default=256, help="Target image size (default: 256)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for resized images (default: overwrite originals)",
    )

    args = parser.parse_args()

    resize_images(
        src_dir=args.src_dir,
        src_size=args.src_size,
        target_dir=args.target_dir,
        target_size=args.target_size,
        output_dir=args.output_dir,
    )

    print("✓ Resize complete!")
