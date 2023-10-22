import numpy as np
# Compute the prediction with onnxruntime.
import onnxruntime as rt
import timeit

X_test = np.random.rand(100, 100)

with open("linear.onnx", "rb") as fp:
    str_ = fp.read()

def unserialize_predict():
    sess = rt.InferenceSession(str_, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    sess.run([label_name], {input_name: X_test})[0]

result = timeit.timeit(unserialize_predict, number=10000)
print(result)