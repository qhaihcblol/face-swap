#!/usr/bin/env python3
"""
Export BlendSwap to ONNX.

Three models are exported:
  identity_extractor.onnx  — source_img [B,3,112,112] → z_id [B,512]   (BlendFace encoder)
  blend_decoder.onnx       — (target_img [B,3,256,256], z_id [B,512]) → output [B,3,256,256]
  blendswap_full.onnx      — (target_img [N,3,256,256], source_img [1,3,112,112]) → output [N,3,256,256]

Note: blendswap_full supports multi-target natively — z_id from source broadcasts
to N targets inside the ADD layers, so source always has batch=1.

Usage:
  python export_onnx.py -w src/checkpoints/blendswap.pth
  python export_onnx.py -w src/checkpoints/blendswap.pth --verify
  python export_onnx.py -w src/checkpoints/blendswap.pth -o models/ --opset 17
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from src.models.blendswap import BlendSwap


# ── ONNX wrappers ──────────────────────────────────────────────────────────────

class IdentityExtractorONNX(nn.Module):
    """BlendFace: source_img → normalised z_id"""

    def __init__(self, model: BlendSwap):
        super().__init__()
        self.identity_encoder = model.identity_encoder

    def forward(self, source_img: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.identity_encoder(source_img))


class BlendDecoderONNX(nn.Module):
    """(target_img, z_id) → swapped output. Efficient for multi-target: pass z_id once."""

    def __init__(self, model: BlendSwap):
        super().__init__()
        self.attribute_encoder = model.attribute_encoder
        self.generator = model.generator
        self.mask_head = model.mask_head

    def forward(self, target_img: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:
        features = self.attribute_encoder(target_img)
        output = self.generator(z_id, features)
        mask = self.mask_head(features[-1]).sigmoid()
        return output * mask + target_img * (1 - mask)


# ── Export helpers ─────────────────────────────────────────────────────────────

def _export(wrapper, out_path, dummy_inputs, input_names, output_names, dynamic_axes, opset):
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(out_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    ins = ", ".join(f"{n}:{d}" for n, d in zip(input_names, [i.shape for i in dummy_inputs]))
    print(f"  [OK] {out_path.name:<30}  {size_mb:6.1f} MB  |  {ins} -> {output_names[0]}")


def export_identity_extractor(model, out_dir, opset, device):
    wrapper = IdentityExtractorONNX(model).eval().to(device)
    dummy = torch.randn(1, 3, 112, 112, device=device)
    _export(
        wrapper, out_dir / "identity_extractor.onnx",
        dummy_inputs=(dummy,),
        input_names=["source_img"],
        output_names=["z_id"],
        dynamic_axes={"source_img": {0: "batch"}, "z_id": {0: "batch"}},
        opset=opset,
    )


def export_blend_decoder(model, out_dir, opset, device):
    wrapper = BlendDecoderONNX(model).eval().to(device)
    dummy_target = torch.randn(1, 3, 256, 256, device=device)
    dummy_z_id = torch.randn(1, 512, device=device)
    _export(
        wrapper, out_dir / "blend_decoder.onnx",
        dummy_inputs=(dummy_target, dummy_z_id),
        input_names=["target_img", "z_id"],
        output_names=["output"],
        dynamic_axes={"target_img": {0: "batch"}, "z_id": {0: "batch"}, "output": {0: "batch"}},
        opset=opset,
    )


def export_full(model, out_dir, opset, device):
    # BlendSwap.forward is already clean — export directly.
    # source stays batch=1; target batch is dynamic (N targets, 1 source).
    wrapper = model.eval().to(device)
    dummy_target = torch.randn(1, 3, 256, 256, device=device)
    dummy_source = torch.randn(1, 3, 112, 112, device=device)
    _export(
        wrapper, out_dir / "blendswap_full.onnx",
        dummy_inputs=(dummy_target, dummy_source),
        input_names=["target_img", "source_img"],
        output_names=["output"],
        dynamic_axes={"target_img": {0: "n_targets"}, "output": {0: "n_targets"}},
        opset=opset,
    )


# ── Verification ───────────────────────────────────────────────────────────────

def _ort_run(onnx_path, feed):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def _check(name, pt_out, ort_out, tol=1e-4):
    max_diff = float(np.abs(pt_out - ort_out).max())
    status = "OK" if max_diff < tol else "FAIL"
    print(f"  [{status}] {name:<28}  max_diff={max_diff:.2e}")


def verify_exports(model, out_dir, device, skip_full):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("\n[SKIP] onnxruntime not installed — run: pip install onnxruntime")
        return

    print("\nVerifying ONNX vs PyTorch (tol=1e-4) ...")
    torch.manual_seed(42)
    t_img = torch.randn(1, 3, 256, 256, device=device)
    s_img = torch.randn(1, 3, 112, 112, device=device)

    with torch.inference_mode():
        z_id_pt = F.normalize(model.identity_encoder(s_img)).cpu().numpy()
        _check("identity_extractor",
               z_id_pt,
               _ort_run(out_dir / "identity_extractor.onnx", {"source_img": s_img.cpu().numpy()}))

        z_id_t = torch.from_numpy(z_id_pt).to(device)
        features = model.attribute_encoder(t_img)
        dec_pt = (model.generator(z_id_t, features) * model.mask_head(features[-1]).sigmoid()
                  + t_img * (1 - model.mask_head(features[-1]).sigmoid())).cpu().numpy()
        _check("blend_decoder",
               dec_pt,
               _ort_run(out_dir / "blend_decoder.onnx",
                        {"target_img": t_img.cpu().numpy(), "z_id": z_id_pt}))

        if not skip_full:
            full_pt = model(t_img, s_img).cpu().numpy()
            _check("blendswap_full",
                   full_pt,
                   _ort_run(out_dir / "blendswap_full.onnx",
                            {"target_img": t_img.cpu().numpy(), "source_img": s_img.cpu().numpy()}))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-w", "--weight_path", required=True,
                   help="Path to blendswap .pth checkpoint")
    p.add_argument("-o", "--output_dir", default="onnx_models",
                   help="Output directory  (default: onnx_models)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version  (default: 17)")
    p.add_argument("--no_full", action="store_true",
                   help="Skip exporting blendswap_full.onnx")
    p.add_argument("--verify", action="store_true",
                   help="Verify ONNX outputs against PyTorch (requires onnxruntime)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.weight_path}")
    model = BlendSwap.from_checkpoint(args.weight_path).to(args.device)

    print(f"\nExporting to {out_dir}/  (opset={args.opset}, device={args.device})")
    export_identity_extractor(model, out_dir, args.opset, args.device)
    export_blend_decoder(model, out_dir, args.opset, args.device)
    if not args.no_full:
        export_full(model, out_dir, args.opset, args.device)

    if args.verify:
        verify_exports(model, out_dir, args.device, args.no_full)


if __name__ == "__main__":
    main()
