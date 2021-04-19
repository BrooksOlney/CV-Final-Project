import numpy as np
import tensorflow as tf
from time import time

def test(model, x, y):
    start = time()
    preds = model.predict(x)
    runtime = time() - start

    return runtime, np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

def tflite_test(interpreter, x, y, batchSize = 32):
    start = time()

    input_tensor = interpreter.get_input_details()
    output_tensor = interpreter.get_output_details()
    preds = []

    batches = len(x) // batchSize

    if batches * batchSize < len(x):
        batches += 1
    interpreter.resize_tensor_input(input_tensor[0]['index'], [batchSize, *x[0].shape])
    interpreter.allocate_tensors()

    for i in range(batches):

        batch = x[i*batchSize : (i+1) * batchSize]

        interpreter.set_tensor(input_tensor[0]['index'], batch)
        interpreter.invoke()

        preds.extend(interpreter.get_tensor(output_tensor[0]['index']))

    preds = np.argmax(np.array(preds), axis=1)
    y = np.argmax(y, axis=1)
    runtime = time() - start

    return runtime, np.mean(preds == y)