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

import models.mnist
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing.image #import ImageDataGenerator
from tensorflow.keras.models import Model
from kerassurgeon import Surgeon, identify
from kerassurgeon.operations import delete_channels, delete_layer
import math



def get_filter_weights(model, layer=None):
    """function to return weights array for one or all conv layers of a Keras model"""

    weights = [model.layers[layer_ix].get_weights()[0] for layer_ix in range(len(model.layers))\
    if 'conv' in model.layers[layer_ix].name]
    weight_array = [np.array(i) for i in weights]
    
    return weight_array 

def get_filters_l1(model, layer=None):
    """Returns L1 norm of a Keras model filters at a given conv layer, if layer=None, returns a matrix of norms model is a Keras model"""

    weights = get_filter_weights(model)
    max_kernels = max([layr.shape[3] for layr in weights])
    norms = np.empty((len(weights), max_kernels))
    norms[:] = np.NaN
    for layer_ix in range(len(weights)):
        # compute norm of the filters
        kernel_size = weights[layer_ix][:,:,:,0].size
        nb_filters = weights[layer_ix].shape[3]
        kernels = weights[layer_ix]
        l1 = [np.sum(abs(kernels[:,:,:,i])) for i in range(nb_filters)]
        # divide by shape of the filters
        l1 = np.array(l1) / kernel_size
        norms[layer_ix, :nb_filters] = l1

    return norms

#function to return pruned filters with l1 method
def prune_l1(model, n_pruned, layer=None):
    """returns list of indexes of filter to prune or a matrix layer X filter to prune"""

    norms = get_filters_l1(model)
    to_prune = smallest_indices(norms, n_pruned)
    
    print(f'to_prune {type(to_prune)}')
    
    return to_prune



#Perc parameter sets the percentage of filters in the network to prune. Returns n_pruned which is the number of filters pruned  
def compute_pruned_count(model, perc=0.1, layer=None):
    if layer or layer ==0:
        # count nb of filters
        nb_filters = model.layers[layer].output_shape[3]
    else:
        nb_filters = np.sum([model.layers[i].output_shape[3] for i, layer in enumerate(model.layers) 
                                if 'conv' in model.layers[i].name])
            
    n_pruned = int(np.floor(perc*nb_filters))
    return n_pruned


def smallest_indices(array, N):
    idx = array.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, array.shape)).T

def biggest_indices(array, N):
    idx = array.ravel().argsort()[::-1][:N]
    return np.stack(np.unravel_index(idx, array.shape)).T

# The following code performs actual pruning on the model 
def prune_one_layer(model, pruned_indexes, layer_ix, opt):
    """Prunes one layer based on a Keras Model, layer index 
    and indexes of filters to prune"""
    model_pruned = delete_channels(model, model.layers[layer_ix], pruned_indexes)
    model_pruned.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
    return model_pruned

def prune_multiple_layers(model, pruned_matrix, opt):
    """Prunes several layers based on a Keras Model, layer index and matrix 
    of indexes of filters to prune"""
    conv_indexes = [i for i, v in enumerate(model.layers) if 'conv' in v.name]
    layers_to_prune = np.unique(pruned_matrix[:,0])
    surgeon = Surgeon(model, copy=True)
    to_prune = pruned_matrix
    to_prune[:,0] = np.array([conv_indexes[i] for i in to_prune[:,0]])
    layers_to_prune = np.unique(to_prune[:,0])
    for layer_ix in layers_to_prune :
        pruned_filters = [x[1] for x in to_prune if x[0]==layer_ix]
        pruned_layer = model.layers[layer_ix]
        surgeon.add_job('delete_channels', pruned_layer, channels=pruned_filters)
    
    model_pruned = surgeon.operate()
    model_pruned.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    return model_pruned

def prune_model(model, perc, opt, layer=None):
    """Prune a Keras model using different methods
    Arguments:
        model: Keras Model object
        perc: a float between 0 and 1
    Returns:
        A pruned Keras Model object
    
    """
      
    n_pruned = compute_pruned_count(model, perc, layer)
    
    #to_prune = prune(model, n_pruned, layer)      

    to_prune = prune_l1(model, n_pruned, opt)
            
    model_pruned = prune_multiple_layers(model, to_prune, opt)

    return model_pruned

