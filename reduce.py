"""
    Computer Vision Final Project
    Model reduction techniques for CNNs targeting mobile applications

    Authors:
    Brooks Olney, Peter Stilian

    This is the main script that implements and evaluates the reduction methods.
    Pipeline is as follows:
        1) load model and dataset
        2) compute baseline metrics for accuracy, model size, and performance (time for inference)
        3) perform model reduction technique - pruning or rank factorization
      **4) compute (2) for reduced model
        5) quantize the weights to smaller precision
        6) compute (2) again
        7) plot results

        ** may only compute these metrics once, and compare to baseline for final (post quantization) model
"""

import mnist
import numpy as np

def low_rank_factorization(model):

    for layer in model.layers:
        if "dense" in layer.name:
            newLayers = lrf_fc_layer(layer)


def lrf_fc_layer(layer):

    w, b = layer.get_weights()
    t = w.shape[0]

    U, S, V = np.linalg.svd(w)

    Ut = U[:, :t]
    St = S[:t]
    Vt = V[:t, :]

    L = np.dot(np.diag(St), Vt)



    print("hi")




if __name__ == "__main__":
    model = mnist.build_model()
    (trainX, trainY), (testX, testY) = mnist.load_data()
    low_rank_factorization(model)

    print(model.summary())