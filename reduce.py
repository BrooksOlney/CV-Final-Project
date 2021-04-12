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
import tensorflow as tf

def test(model, x, y):

    preds = model.predict_on_batch(x)

    return np.mean(np.argmax(preds, axis=1) == np.argmax(y, axis=1))

def low_rank_factorization(model):

    newLayers = []

    for i, layer in enumerate(model.layers):

        # apply lra to every FC layer except the output layer
        if "dense" in layer.name and i < len(model.layers) - 1:
            ls = lrf_fc_layer(layer)
            newLayers.extend(ls)
        else:
            newLayers.append(layer)

    newModel = tf.keras.models.Sequential()

    for layer in newLayers:
        newModel.add(layer)

    newModel.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return newModel

def lrf_fc_layer(layer, t=20):

    w, b = layer.get_weights()

    U, S, V = np.linalg.svd(w, full_matrices=False)

    t = r = np.linalg.matrix_rank(w)
    t = 20

    Ut = U[:, :t]
    St = S[:t]
    Vt = V[:t, :]

    L = np.dot(np.diag(St), Vt)

    l1 = tf.keras.layers.Dense(L.shape[1], activation=layer.activation)
    l1.build(L.shape[0])
    l1.set_weights([L, np.zeros(L.shape[1],dtype=np.float32)])
    # l1.weights = [L, np.zeros(L.shape[0],dtype=np.float32)]

    l2 = tf.keras.layers.Dense(Ut.shape[1])
    l2.build(Ut.shape[::-1])
    l2.set_weights([Ut, b[:t]])

    return l2, l1


if __name__ == "__main__":
    model = mnist.build_model()
    print(model.summary())
    (trainX, trainY), (testX, testY) = mnist.load_data()
    newModel = low_rank_factorization(model)
    acc = test(newModel, testX, testY)
    newModel.fit(trainX, trainY, 128, 10)
    acc = test(newModel, testX, testY)

    results = newModel.evaluate(testX, testY)
    print("test loss, test acc:", results)
    
    print(newModel.summary())

