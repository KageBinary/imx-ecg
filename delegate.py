import tflite_runtime.interpreter as tflite
import numpy as np

model_path = '/root/imx-ecg/artifacts/ecg_deploy_int8.tflite'

interp = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('/usr/lib/libvx_delegate.so')]
)
interp.allocate_tensors()

print("=== All ops and their delegation ===")
for i, op in enumerate(interp._get_ops_details()):
    print(f"Op {i:3d}: {op['op_name']}")