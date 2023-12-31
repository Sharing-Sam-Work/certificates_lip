# v2 can handle calculating certificates on a batch of inputs as opposed to a single input
# v3 can handle activation layers/functions
# v4 expands on v3, by handling more activation layers/functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers# , Models
import numpy as np
import logging
import deel
from deel import lip

# currently, this code does not take into account the activation layers/functions.

logging.basicConfig(level=logging.WARNING)

GLOBAL_CONSTANTS = {
    "supported_neutral_layers": ["flatten", "input", "activation", "p_re_lu"], 
   
    "not_deel": [
        "dense",
        "average_pooling2d",
        "global_average_pooling2d",
        "conv2d"
    ],
    "supported_Lipschitz_layers": [
        "spectral_dense",
        "frobenius_dense",
        "scaled_average_pooling2d",
        "scaled_l2_norm_pooling2d",
        "scaled_global_average_pooling2d",
        "spectral_conv2d",
        "frobenius_conv2d",
        "group_sort2",
        "full_sort",
        "group_sort",
        "householder",
        "max_min"
        
    ],
    "not_Lipschitz": [
        "batch_normalization", 
        "dropout",
        "leaky_re_lu",
        "elu",
        "thresholded_re_lu"
    ],

    

    "unrecommended_activation_functions": [tf.keras.activations.relu, tf.keras.activations.softmax,\
                                          tf.keras.activations.exponential,tf.keras.activations.elu,\
                            tf.keras.activations.selu,tf.keras.activations.tanh, \
                            tf.keras.activations.sigmoid, tf.keras.activations.softplus, tf.keras.activations.softsign],
    "min_max_norm":keras.constraints.MinMaxNorm,
    "recommended_activation_names": ['group_sort2', 'full_sort', 'group_sort', 'householder', 'max_min', 'p_re_lu'],
    "unrecommended_activation_names": ['re_lu'],
    "no_activation": tf.keras.activations.linear ,
    
}

def get_layers(model):
    return model.layers

class NotLipschtzLayerError(Exception):
    pass
class BadLastLayerError(Exception):
    pass

def check_is_Lipschitz(layers):
    """
    
    """
    for i,layer in enumerate(layers):
        check_activation_layer(layer)
        if any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["supported_neutral_layers"]):
            # print("layer neutral")
            pass
        elif any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["supported_Lipschitz_layers"]):
            pass 
        elif any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["not_Lipschitz"]):
            # triggers when using none Lipschitz layers such as "batch_normalization"
            raise NotLipschtzLayerError("The layer '%s' is not supported" %layer.name)
        elif any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["not_deel"]):
            #print("Layer %s not deel." %layer.name)
            logging.warning("A deel equivalent exists for  '%s'. For practical purposes, we will assume that the layer is 1-Lipschitz." %layer.name)
        elif any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["unrecommended_activation_names"]):
            # triggers when using tf.keras.layers.ReLU()
            logging.warning("The layer '%s' is not recommended. \
For practical purposes, we recommend to use deel lip activation layer instead such as GroupSort2.\n" %(layer.name))
        else:
            
            logging.warning("Unknown layer '%s' used. For practical purposes, we will assume that the layer is 1-Lipschitz." %layer.name)

def check_activation_layer(layer):
    """
    
    """
    if hasattr(layer, 'activation'):
        
        activation = getattr(layer, 'activation')
        
        if activation!=GLOBAL_CONSTANTS["no_activation"]:

            
            if activation in GLOBAL_CONSTANTS["unrecommended_activation_functions"]:
                # triggers when calling unrecommended activation function e.g. 
                # lip.layers.SpectralDense(64, activation='selu')(x)
                logging.warning("The '%s' activation function of the layer '%s' is not recommended. \
For practical purposes, we recommend to use deel lip activation functions instead such as GroupSort2.\n" %(activation,layer.name))
                return None
            
            if isinstance(activation, GLOBAL_CONSTANTS["min_max_norm"]):
                return None
            elif hasattr(activation, 'name'):
                
                n=activation.name
                # print(n)
                if any(n.startswith(substring) for substring in GLOBAL_CONSTANTS["recommended_activation_names"]):
                    return None
                elif any(n.startswith(substring) for substring in GLOBAL_CONSTANTS["unrecommended_activation_names"]):
                    logging.warning("The '%s' activation function of the layer '%s' is not recommended. \
For practical purposes, we recommend to use deel lip activation functions instead such as GroupSort2.\n" %(activation,n))
                else:
                    print("The '%s' activation function of the layer '%s' is unknown. We will assume it is 1-Lipschitz.\n" %(n,layer.name))

            else:
                
                logging.warning("The '%s' activation function of the layer '%s' is unknown.\n" %(activation,layer.name))






def get_K_(layers):
    try:
        check_is_Lipschitz(layers)
    except NotLipschtzLayerError as e:
        print("NotLipschtzLayerError raised:", e)
        return None
    
    K_ = 1
    # Print information about each layer
    for layer in layers:
        if any(layer.name.startswith(substring) for substring in GLOBAL_CONSTANTS["supported_Lipschitz_layers"]):
            K_ = layer.k_coef_lip * K_
        else:
            pass
    return K_

def get_last_layer(model):
    return get_layers(model)[-1]

def get_weights_last_layer(model, check=True):
    last_layer=get_last_layer(model)

    if check:
        # check last layer is a layer with weights
        if not hasattr(last_layer,'get_weights'):
             raise BadLastLayerError("The last layer '%s' must have a set of weights to calculate the certificate." %last_layer.name)
        # check last layer has no activation function set
        activation = getattr(last_layer, 'activation')
        if activation!=GLOBAL_CONSTANTS["no_activation"]:
            logging.warning("We recommend avoiding using an activation \
function for the last layer (here the '%s' activation function of the layer '%s').\n"%(activation,last_layer.name))
    
    return get_last_layer(model).get_weights()[0]

def get_sorted_logits_indices(model_output):
    # Sort the model outputs model.predict(x)
    sorted_indices = np.argsort(model_output, axis=1)[:, ::-1]
    return sorted_indices


def get_num_perceptron_last_layer(model):
    return get_weights_last_layer(model).shape[1]

def get_certificate(model,x):
    num_perceptron_last_layer = get_num_perceptron_last_layer(model)
    if num_perceptron_last_layer>=3:
        return get_certificate_multiclassification(model, x, num_classes = num_perceptron_last_layer)
    elif num_perceptron_last_layer==1:
        return get_certificate_binary(model,x)
    else:
        print("You have 2 perceptrons in the last layer./n \
        The prefered approach for binary classification is to have one perceptron.")



def get_certificate_multiclassification(model, x, num_classes):

    certificate=[]
    
    min_value = float('inf')  # Initialize with positive infinity


    model_output =  model.predict(x)


    last_layer_weights =  get_weights_last_layer(model, False)

    
    K_n_minus_1 = get_K_(get_layers(model)[:-1])

    if K_n_minus_1 is None:
        return None

    sorted_indices = get_sorted_logits_indices(model_output)
    for j in range(len(model_output)):
        min_value = float('inf')
        for i in range(1, num_classes):
            numerator = model_output[j][sorted_indices[j][0]] - model_output[j][sorted_indices[j][i]]
            denominator = np.linalg.norm(last_layer_weights[:, sorted_indices[j][0]] - last_layer_weights[:, sorted_indices[j][i]])
            formula_value = numerator / (denominator * K_n_minus_1)
            min_value = min(min_value, formula_value)
        certificate.append(min_value)
        
        
    return np.array(certificate)

def get_certificate_binary(model,x):
    K_= get_K_(get_layers(model))
    if K_ is None:
        return None
    model_output = model.predict(x)
    return np.abs(model_output[:, 0]) / K_

