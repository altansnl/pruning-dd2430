import tensorflow as tf
from keras.saving.save import load_model
from tensorflow import keras
import numpy as np
from model import CVAE

def structural_prune(
    cvae: CVAE, 
    rewind_cvae: CVAE, 
    scenario, 
    left_to_prune_encoder=None, 
    left_to_prune_decoder=None,
    final_prune_percentage=0.8,
    prunes_left=5
    ):
    if scenario == 1:
        pruned_cvae = cvae
    elif scenario == 2:
        pruned_encoder, left_to_prune_encoder = _structural_prune_submodel(cvae.encoder, rewind_cvae.encoder, left_to_prune_encoder, final_prune_percentage, prunes_left)
        pruned_encoder.save('saved_models/pruned_encoder.h5')
        pruned_encoder = load_model('saved_models/pruned_encoder.h5', compile=False)
        pruned_decoder = cvae.decoder
        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)

    elif scenario == 3:
        pruned_encoder = cvae.encoder
        pruned_decoder, left_to_prune_decoder = _structural_prune_submodel(cvae.decoder, rewind_cvae.decoder, left_to_prune_decoder, final_prune_percentage, prunes_left)
        pruned_decoder.save('saved_models/pruned_decoder.h5')
        pruned_decoder = load_model('saved_models/pruned_decoder.h5')
        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)

    elif scenario == 4:
        pruned_encoder, left_to_prune_encoder = _structural_prune_submodel(cvae.encoder, rewind_cvae.encoder, left_to_prune_encoder, final_prune_percentage, prunes_left)
        pruned_encoder.save('saved_models/pruned_encoder.h5')
        pruned_encoder = load_model('saved_models/pruned_encoder.h5')
        pruned_decoder, left_to_prune_decoder = _structural_prune_submodel(cvae.decoder, rewind_cvae.decoder, left_to_prune_decoder, final_prune_percentage, prunes_left)
        pruned_decoder.save('saved_models/pruned_decoder.h5')
        pruned_decoder = load_model('saved_models/pruned_decoder.h5')
        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)
    else:
        raise ValueError("Not applicable Scenario.")
    return pruned_cvae, left_to_prune_encoder, left_to_prune_decoder


def _structural_prune_submodel(
    submodel: tf.keras.Sequential, 
    rewind_submodel: tf.keras.Sequential, 
    left_to_prune=None, 
    final_prune_percentage=0.8,
    prunes_left=5
    ):
    # A map keeping how many filters left to be removed per prunable layer
    if left_to_prune is None:
        left_to_prune_internal = {}
    else:
        left_to_prune_internal = left_to_prune

    # Add input layer
    input_layer = tf.keras.layers.InputLayer(input_shape=submodel.layers[0].input_shape[1:])
    pruned_submodel = tf.keras.Sequential([input_layer])
    remaining_filter_indices_list = []
    for no, layer in enumerate(submodel.layers[:-1]):  # Exclude last layer
        if isinstance(layer, keras.layers.Conv2D):
            conv_weights, _ = layer.get_weights()
            if isinstance(layer, keras.layers.Conv2DTranspose):
                kernel_sum = np.sum(np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=-1)
            else:
                kernel_sum = np.sum(np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=0)

            if left_to_prune is None:
                left_to_prune_internal[no] = int(len(kernel_sum) * final_prune_percentage) # total number of filters to remove from this layer
                
            n_filters2remove = int(left_to_prune_internal[no] / prunes_left)
            remaining_filter_indices = np.sort(np.argsort(kernel_sum)[n_filters2remove:])
            remaining_filter_indices_list.append(remaining_filter_indices)
            
            # update how many filters are left to prune for this layer
            left_to_prune_internal[no] -= n_filters2remove

            layer_config = layer.get_config()
            layer_config['filters'] = len(remaining_filter_indices)
            new_layer = type(layer).from_config(layer_config)
            pruned_submodel.add(new_layer)
            # Get rewind weights
            rewind_weights, rewind_biases = rewind_submodel.layers[no].get_weights()

            if isinstance(submodel.layers[no - 1], keras.layers.Conv2D):
                previous_layer_remaining_filter_indices = remaining_filter_indices_list[-2]
                if isinstance(submodel.layers[no - 1], keras.layers.Conv2DTranspose):
                    updated_rewind_weights = rewind_weights[:, :, :, previous_layer_remaining_filter_indices]
                else:
                    updated_rewind_weights = rewind_weights[:, :, previous_layer_remaining_filter_indices, :]
                if isinstance(layer, keras.layers.Conv2DTranspose):
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :, remaining_filter_indices, :],
                        rewind_biases[remaining_filter_indices]
                    ])
                else:
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :, :, remaining_filter_indices],
                        rewind_biases[remaining_filter_indices]
                    ])

            else:
                if isinstance(layer, keras.layers.Conv2DTranspose):
                    pruned_submodel.layers[-1].set_weights([
                        rewind_weights[:, :, remaining_filter_indices, :],
                        rewind_biases[remaining_filter_indices]
                    ])

                else:
                    pruned_submodel.layers[-1].set_weights([
                        rewind_weights[:, :, :, remaining_filter_indices],
                        rewind_biases[remaining_filter_indices]
                    ])
        else:
            config = rewind_submodel.layers[no].get_config()
            weights = rewind_submodel.layers[no].get_weights()
            cloned_layer = type(rewind_submodel.layers[no]).from_config(config)
            pruned_submodel.add(cloned_layer)
            pruned_submodel.layers[-1].set_weights(weights)

    # Add last layer
    if isinstance(submodel.layers[-1], keras.layers.Conv2D):
        output_layer_config = submodel.layers[-1].get_config()
        output_layer = type(submodel.layers[-1]).from_config(output_layer_config)
        pruned_submodel.add(output_layer)
        # Get rewind weights
        rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights()
        if isinstance(submodel.layers[-2], keras.layers.Conv2D):
            previous_layer_remaining_filter_indices = remaining_filter_indices_list[-1]
            if isinstance(submodel.layers[-2], keras.layers.Conv2DTranspose):
                updated_rewind_weights = rewind_weights[:, :, :, previous_layer_remaining_filter_indices]
            else:
                updated_rewind_weights = rewind_weights[:, :, previous_layer_remaining_filter_indices, :]
            pruned_submodel.layers[-1].set_weights([
                updated_rewind_weights,
                rewind_biases
            ])
        else:
            pruned_submodel.layers[-1].set_weights([
                rewind_weights,
                rewind_biases
            ])
    elif isinstance(submodel.layers[-1], keras.layers.Dense):
        if not isinstance(submodel.layers[-2], keras.layers.Dense) or not isinstance(submodel.layers[-2], keras.layers.Conv2D):
            if isinstance(submodel.layers[-3], keras.layers.Conv2D):
                previous_layer_remaining_filter_indices = remaining_filter_indices_list[-1]
                output_layer_config = rewind_submodel.layers[-1].get_config()
                output_layer = type(rewind_submodel.layers[-1]).from_config(output_layer_config)
                pruned_submodel.add(output_layer)
                rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights()
                updated_rewind_weights = rewind_weights[previous_layer_remaining_filter_indices, :]
                pruned_submodel.layers[-1].set_weights([
                    updated_rewind_weights,
                    rewind_biases
                ])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        output_layer_config = rewind_submodel.layers[-1].get_config()
        weights = rewind_submodel.layers[-1].get_weights()
        output_layer = type(rewind_submodel.layers[-1]).from_config(output_layer_config)
        pruned_submodel.add(output_layer)
        pruned_submodel.layers[-1].set_weights(weights)

    return pruned_submodel, left_to_prune_internal

def restore_layer_names(layer, pruned_submodel):
    layer_config = layer.layers[-1].get_config()
    layer_config['name'] = 'pruned_' + layer_config['name']
    new_layer = type(layer).from_config(layer_config)
    if pruned_submodel.layers[-2].name.startswith('pruned_'):
        pass  # adjust connections
    raise NotImplementedError
