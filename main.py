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

# some debug info only relevant to my machine
# RTX = DEBUG = os.environ['COMPUTERNAME'] == 'BROOKSRIG'
loc  = "F:\\Coursework\\Spring2021\\ComputerVision\\FinalProject\\"

from models import mnist, vggcifar10
import pruning_tools as pt
import reduction_tools as rt
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee'])


if __name__ == "__main__":

    curArch = vggcifar10
    
    model = curArch.build_model()
    (trainX, trainY), (testX, testY) = curArch.load_data()

    xs = np.arange(0.02, 0.22, 0.02)

    accs = []
    runtimes = []
    sizes = []

    for x in xs:
        step1 = pt.prune_model(model, x, tf.keras.optimizers.RMSprop())
        step2 = rt.low_rank_factorization(step1, t=20)

        finalModel = rt.quantize_weights(step2)
        size = len(finalModel._model_content) / 1024 / 1024

        # with open(loc + f"{curArch.__name__}.tflite", "wb") as out:
        #     out.write(finalModel._model_content)

        runtime,acc = utils.tflite_test(finalModel, testX, testY, batchSize=100)
        # print(step2.summary())
        accs.append(acc)
        runtimes.append(runtime)
        sizes.append(size)
    
    color1 = 'tab:blue'
    fig, ax = plt.subplots()
    ax.set_ylabel('Accuracy (\%)', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    
    ax.plot(xs * 100, np.array(accs) * 100, color=color1)

    color2 = 'tab:red'
    ax2 = ax.twinx()
    ax2.set_ylabel('Model Size (KB)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax2.plot(xs * 100, np.array(sizes), color=color2)
    plt.xlabel("\% Neurons Pruned")
    plt.title("VGG-16 + CIFAR-10 Classifier Compression")
    fig.tight_layout()
    plt.savefig('vgg.pdf')
    plt.show()

    print(runtime, acc)

    # runtime,acc = utils.test(model, testX, testY)
    # print(model.summary())
    # print(runtime, acc)