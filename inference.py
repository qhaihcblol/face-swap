import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.blendswap import BlendSwap

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight_path", required=True)
parser.add_argument("-t", "--target_images", nargs="+", default=["examples/target.png"])
parser.add_argument("-s", "--source_image", default="examples/source.png")
parser.add_argument("-o", "--output_dir", default="examples/output")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlendSwap.from_checkpoint(args.weight_path).to(device)

to_tensor = transforms.ToTensor()

source_img = (
    to_tensor(Image.open(args.source_image).convert("RGB").resize((112, 112)))
    .unsqueeze(0)
    .to(device)
)
target_imgs = torch.stack([
    to_tensor(Image.open(p).convert("RGB").resize((256, 256)))
    for p in args.target_images
]).to(device)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with torch.inference_mode():
    outputs = model(target_imgs, source_img)  # [N, 3, 256, 256]

for output, target_path in zip(outputs, args.target_images):
    arr = (output.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    out_path = output_dir / f"{Path(target_path).stem}_swapped.png"
    Image.fromarray(arr).save(out_path)
    print(f"Saved: {out_path}")
