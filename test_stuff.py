import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(
    model_path='/root/imx-ecg/artifacts/ecg_deploy_int8.tflite',
    experimental_delegates=[tflite.load_delegate('/usr/lib/libvx_delegate.so')]
)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()
print('Input:', inp[0]['shape'], inp[0]['dtype'])
print('Output:', out[0]['shape'], out[0]['dtype'])
dummy = np.zeros(inp[0]['shape'], dtype=np.float32)
interpreter.set_tensor(inp[0]['index'], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out[0]['index'])
print('Output logits:', result)
print('Predicted class:', result.argmax())
print('Classes: 0=Normal 1=AF 2=Other 3=Noisy')
