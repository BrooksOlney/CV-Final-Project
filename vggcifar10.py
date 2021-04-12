import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
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

def build_model(filename="models\\cifar10vgg.h5"):
    """
        Builds a model of the VGG-16 architecture and loads weights from pretrained model.
        Returns a keras Sequential object.
    """
    
    model = Sequential()
    
    weight_decay = 0.0005
    num_classes = 10

    model.add(Conv2D(64, (3, 3), padding='same',
                    input_shape=(32, 32, 3),kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
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
        Loads and preprocesses the CIFAR-10 dataset.
        Returns tuples of the training and testing data.
    """

    (trainX, trainY), (testX, testY) = tf.keras.datasets.cifar10.load_data()
    trainX = trainX.reshape(-1, 32, 32, 1).astype(np.float32) / 255
    testX  = testX.reshape(-1, 32, 32, 1).astype(np.float32) / 255

    trainY = tf.keras.utils.to_categorical(trainY)
    testY  = tf.keras.utils.to_categorical(testY)

    mean = np.mean(trainX, axis=(0,1,2,3))
    std  = np.std(trainX, axis=(0,1,2,3))

    trainX = (trainX - mean) / (std + 1e-7)
    testX  = (testX - mean) / (std + 1e-7)

    return (trainX, trainY), (testX, testY)

if __name__ == "__main__":
    vggcifar10model = build_model()
    (trainX, trainY), (testX, testY)  = load_data()

    print(vggcifar10model.summary())
    loss, acc = vggcifar10model.evaluate(testX, testY)
    print("Model performance on test set:\n")
    print("Loss = {}\tAccuracy = {}".format(loss, acc))