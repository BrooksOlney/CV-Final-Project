import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# some debug info only relevant to my machine
RTX = DEBUG = os.environ['COMPUTERNAME'] == 'BROOKSRIG'
DLOC  = "F:\\Coursework\\Spring2021\\ComputerVision\\FinalProject\\"

# tensorflow errors out on my RTX 3090 when trying to allocate too much memory, this limits it
if RTX:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

def build_model(filename="models\\lenet5-mnist.h5"):
    """
        Builds a model of the LeNet-5 architecture and loads weights from pretrained model.
        Returns a keras Sequential object.
    """
    
    model = Sequential()
    
    model.add(Conv2D(6, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    if DEBUG:
        filename = DLOC + filename

    model.load_weights(filename)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

def load_data():
    """
        Loads and preprocesses the MNIST dataset.
        Returns tuples of the training and testing data.
    """

    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
    trainX = trainX.reshape(-1, 28, 28, 1).astype(np.float32) / 255
    testX  = testX.reshape(-1, 28, 28, 1).astype(np.float32) / 255

    trainY = tf.keras.utils.to_categorical(trainY)
    testY  = tf.keras.utils.to_categorical(testY)

    return (trainX, trainY), (testX, testY)

if __name__ == "__main__":
    mnistModel = build_model()
    (trainX, trainY), (testX, testY)  = load_data()

    print(mnistModel.summary())
    loss, acc = mnistModel.evaluate(testX, testY)
    print("Model performance on test set:\n")
    print("Loss = {}\tAccuracy = {}".format(loss, acc))