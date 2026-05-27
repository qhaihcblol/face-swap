import argparse
from pathlib import Path

import cv2
import numpy as np

from src.models.blendswap_face_fusion import BlendSwap

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", default=None,
                    help="Path to blendswap_256.onnx; auto-downloads from HuggingFace if omitted")
parser.add_argument("-t", "--target_images", nargs="+", default=["examples/target.png"])
parser.add_argument("-s", "--source_image", default="examples/source.png")
parser.add_argument("-o", "--output_dir", default="examples/output")
args = parser.parse_args()

model = BlendSwap(model_path=args.model_path)

src_raw = cv2.imread(args.source_image)
if src_raw is None:
    raise FileNotFoundError(args.source_image)
source = cv2.resize(src_raw, (256, 256))

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

for target_path in args.target_images:
    tgt_raw = cv2.imread(target_path)
    if tgt_raw is None:
        print(f"Warning: cannot read {target_path}, skipping")
        continue
    target = cv2.resize(tgt_raw, (256, 256))

    result = model(source, target)  # BGR uint8, 256×256

    out_path = output_dir / f"{Path(target_path).stem}_swapped.png"
    cv2.imwrite(str(out_path), result)
    print(f"Saved: {out_path}")
