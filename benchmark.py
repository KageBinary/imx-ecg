import tflite_runtime.interpreter as tflite
import numpy as np
import time

model_path = '/root/imx-ecg/artifacts/ecg_deploy_int8.tflite'

# With NPU delegate
interp_npu = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('/usr/lib/libvx_delegate.so')]
)
interp_npu.allocate_tensors()
inp = interp_npu.get_input_details()
dummy = np.zeros(inp[0]['shape'], dtype=np.float32)
interp_npu.set_tensor(inp[0]['index'], dummy)
t0 = time.perf_counter()
for _ in range(50):
    interp_npu.invoke()
npu_ms = (time.perf_counter() - t0) / 50 * 1000
# Without delegate (CPU only)
interp_cpu = tflite.Interpreter(model_path=model_path)
interp_cpu.allocate_tensors()
interp_cpu.set_tensor(interp_cpu.get_input_details()[0]['index'], dummy)
t0 = time.perf_counter()
for _ in range(50):
    interp_cpu.invoke()
cpu_ms = (time.perf_counter() - t0) / 50 * 1000

print(f"NPU delegate: {npu_ms:.2f} ms/inference")
print(f"CPU only:     {cpu_ms:.2f} ms/inference")
print(f"Speedup:      {cpu_ms/npu_ms:.1f}x")