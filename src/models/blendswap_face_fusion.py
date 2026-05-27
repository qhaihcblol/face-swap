"""
BlendSwap — pure inference wrapper
Input  : source crop (112×112) + target crop (256×256), đã align sẵn
Output : swapped crop (256×256)

Model  : blendswap_256.onnx từ facefusion/models-3.0.0 (HuggingFace)

pip install onnxruntime huggingface_hub opencv-python numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

_HF_REPO = "facefusion/models-3.0.0"
_MODEL_FILE = "blendswap_256.onnx"
_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _load(src: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return src
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)
    return img


def _to_tensor(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC → float32 NCHW RGB [-1, 1]"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb / 127.5) - 1.0).transpose(2, 0, 1)[None]


def _from_tensor(t: np.ndarray) -> np.ndarray:
    """float32 NCHW RGB [-1, 1] → BGR uint8 HWC"""
    out = np.clip((t[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class BlendSwap:
    """
    Parameters
    ----------
    model_path : path to blendswap_256.onnx; auto-downloads if None
    providers  : ORT execution providers

    Usage
    -----
    swap = BlendSwap()
    result = swap(source_112, target_256)   # numpy BGR uint8, returns 256×256
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        providers: Optional[list[str]] = None,
    ) -> None:
        if model_path is None:
            model_path = hf_hub_download(repo_id=_HF_REPO, filename=_MODEL_FILE)

        self._sess = ort.InferenceSession(
            str(model_path),
            providers=providers or _PROVIDERS,
        )

        by_name = {inp.name: inp for inp in self._sess.get_inputs()}
        self._src_name = "source"
        self._tgt_name = "target"
        self._out_name = self._sess.get_outputs()[0].name

        self._src_size = tuple(by_name["source"].shape[2:])  # (112, 112)
        self._tgt_size = tuple(by_name["target"].shape[2:])  # (256, 256)

    def __call__(
        self,
        source: str | Path | np.ndarray,
        target: str | Path | np.ndarray,
    ) -> np.ndarray:
        """
        source : BGR uint8, already aligned to 112×112 (ArcFace crop)
        target : BGR uint8, already aligned to 256×256 (FFHQ crop)
        returns: BGR uint8, 256×256 swapped face
        """
        src = _load(source)
        tgt = _load(target)

        assert (
            src.shape[:2] == self._src_size
        ), f"source must be {self._src_size}, got {src.shape[:2]}"
        assert (
            tgt.shape[:2] == self._tgt_size
        ), f"target must be {self._tgt_size}, got {tgt.shape[:2]}"

        out = self._sess.run(
            [self._out_name],
            {self._src_name: _to_tensor(src), self._tgt_name: _to_tensor(tgt)},
        )[0]

        return _from_tensor(out)
