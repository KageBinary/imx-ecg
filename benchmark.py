import tflite_runtime.interpreter as tflite
import numpy as np
import time

model_path = '/root/imx-ecg/artifacts/ecg_deploy_int8.tflite'

def make_dummy(interp):
    inp = interp.get_input_details()[0]
    dummy_f32 = np.zeros(inp['shape'], dtype=np.float32)
    if inp['dtype'] == np.int8:
        scale, zero_point = inp['quantization']
        return (dummy_f32 / scale + zero_point).astype(np.int8)
    return dummy_f32

# With NPU delegate
interp_npu = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('/usr/lib/libvx_delegate.so')]
)
interp_npu.allocate_tensors()
dummy_npu = make_dummy(interp_npu)
interp_npu.set_tensor(interp_npu.get_input_details()[0]['index'], dummy_npu)
t0 = time.perf_counter()
for _ in range(50):
    interp_npu.invoke()
npu_ms = (time.perf_counter() - t0) / 50 * 1000

# Without delegate (CPU only)
interp_cpu = tflite.Interpreter(model_path=model_path)
interp_cpu.allocate_tensors()
dummy_cpu = make_dummy(interp_cpu)
interp_cpu.set_tensor(interp_cpu.get_input_details()[0]['index'], dummy_cpu)
t0 = time.perf_counter()
for _ in range(50):
    interp_cpu.invoke()
cpu_ms = (time.perf_counter() - t0) / 50 * 1000

print(f"NPU delegate: {npu_ms:.2f} ms/inference")
print(f"CPU only:     {cpu_ms:.2f} ms/inference")
print(f"Speedup:      {cpu_ms/npu_ms:.1f}x")
