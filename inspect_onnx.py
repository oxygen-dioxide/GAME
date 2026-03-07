import onnx
import sys
import glob

def print_io(model_path):
    print(f"--- {model_path} ---")
    try:
        model = onnx.load(model_path)
        print("Inputs:")
        for inp in model.graph.input:
            shape = []
            if inp.type.tensor_type.HasField("shape"):
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(str(dim.dim_value))
                    elif dim.HasField("dim_param"):
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            print(f"  {inp.name} : {inp.type.tensor_type.elem_type} {shape}")
        print("Outputs:")
        for out in model.graph.output:
            shape = []
            if out.type.tensor_type.HasField("shape"):
                for dim in out.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(str(dim.dim_value))
                    elif dim.HasField("dim_param"):
                        shape.append(dim.dim_param)
                    else:
                        shape.append("?")
            print(f"  {out.name} : {out.type.tensor_type.elem_type} {shape}")
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
    print()

for p in glob.glob("experiments/GAME-1.0-small-onnx-static/*.onnx"):
    print_io(p)
