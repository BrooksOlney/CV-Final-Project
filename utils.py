import numpy as np
import tensorflow as tf
from time import time

def test(model, x, y):
    start = time()
    preds = model.predict(x)
    runtime = time() - start

    return runtime, np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

    