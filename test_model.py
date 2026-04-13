import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(
    model_path='/root/imx-ecg/artifacts/ecg_deploy_int8.tflite',
    experimental_delegates=[tflite.load_delegate('/usr/lib/libvx_delegate.so')]
)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print('Input  shape:', inp['shape'], 'dtype:', inp['dtype'])
print('Output shape:', out['shape'], 'dtype:', out['dtype'])

# Build dummy input in the correct dtype
# For int8 models: scale float32 input using quantization params
dummy_f32 = np.zeros(inp['shape'], dtype=np.float32)
if inp['dtype'] == np.int8:
    scale, zero_point = inp['quantization']
    dummy = (dummy_f32 / scale + zero_point).astype(np.int8)
else:
    dummy = dummy_f32

interpreter.set_tensor(inp['index'], dummy)
interpreter.invoke()
raw = interpreter.get_tensor(out['index'])

# Dequantize output if int8
if out['dtype'] == np.int8:
    scale, zero_point = out['quantization']
    logits = (raw.astype(np.float32) - zero_point) * scale
else:
    logits = raw.astype(np.float32)

print('Output logits:', logits)
print('Predicted class:', logits.argmax())
print('Classes: 0=Normal 1=AF 2=Other 3=Noisy')
