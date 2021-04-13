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
import vggcifar10
import pruning_tools
import reduce
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
import numpy as np
import os


if __name__ == "__main__":
    model = mnist.build_model()
    print(model.summary())

    # set amount (by percentage) to prune here
    percent = 0.2

    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    (trainX, trainY), (testX, testY) = mnist.load_data()
        
    #Begin pruning Here
    to_prune = pruning_tools.prune_model(model, percent, opt)

    model_pruned = pruning_tools.prune_multiple_layers(model, to_prune, opt)

    print(type(model_pruned))

    #Retest
    acc = reduce.test(model_pruned, testX, testY)
    print(model_pruned.summary())
    results = model_pruned.evaluate(testX, testY)
    print(f"percentage pruned: {percent * 100}%")
    print("test loss, test acc:", results)