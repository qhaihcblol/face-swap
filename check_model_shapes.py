"""
Print input/output shapes of blendswap_256.onnx from HuggingFace.
"""

import onnxruntime as ort
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="facefusion/models-3.0.0", filename="blendswap_256.onnx")
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print("=== INPUTS ===")
for i, inp in enumerate(sess.get_inputs()):
    print(f"  [{i}] name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")

print("=== OUTPUTS ===")
for i, out in enumerate(sess.get_outputs()):
    print(f"  [{i}] name={out.name!r}  shape={out.shape}  dtype={out.type}")
