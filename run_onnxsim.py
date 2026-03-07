import glob
import os
import subprocess

src_dir = r"experiments/GAME-1.0-small-onnx-opset18"
dst_dir = r"experiments/GAME-1.0-small-onnx-sim"

for f in glob.glob(os.path.join(src_dir, "*.onnx")):
    fname = os.path.basename(f)
    out_path = os.path.join(dst_dir, fname)
    print(f"Simplifying {f} to {out_path}")
    subprocess.run(["onnxsim", f, out_path])

print("Done")
