import numpy as np
import tensorflow as tf
from tensorflow import keras

import utils
from model import CVAE


def structural_prune(cvae: CVAE, rewind_cvae: CVAE, m, scenario):
    if scenario == 1:
        utils.save_models(cvae.encoder, cvae.decoder, scenario)
        pruned_cvae = utils.reset_and_load_models(cvae.latent_dim, scenario)

    elif scenario == 2:
        pruned_encoder = _structural_prune_submodel(cvae.encoder, rewind_cvae.encoder, m)

        utils.save_models(pruned_encoder, cvae.decoder, scenario)
        pruned_cvae = utils.reset_and_load_models(cvae.latent_dim, scenario)

    elif scenario == 3:
        pruned_decoder = _structural_prune_submodel(cvae.decoder, rewind_cvae.decoder, m)

        utils.save_models(cvae.encoder, pruned_decoder, scenario)
        pruned_cvae = utils.reset_and_load_models(cvae.latent_dim, scenario)

    elif scenario == 4:
        pruned_encoder = _structural_prune_submodel(cvae.encoder, rewind_cvae.encoder, m)
        pruned_decoder = _structural_prune_submodel(cvae.decoder, rewind_cvae.decoder, m)

        utils.save_models(pruned_encoder, pruned_decoder, scenario)
        pruned_cvae = utils.reset_and_load_models(cvae.latent_dim, scenario)

    else:
        raise ValueError("Not applicable Scenario.")

    return pruned_cvae


def _structural_prune_submodel(submodel: tf.keras.Sequential, rewind_submodel: tf.keras.Sequential, m):
    # Add input layer
    input_layer = tf.keras.layers.InputLayer(input_shape=submodel.layers[0].input_shape[1:])
    pruned_submodel = tf.keras.Sequential([input_layer])

    remaining_filter_indices_list = []
    for no, layer in enumerate(submodel.layers[:-1]):  # Exclude last layer
        if isinstance(layer, keras.layers.Conv2D):
            conv_weights, conv_biases = layer.get_weights()
            if isinstance(layer, keras.layers.Conv2DTranspose):
                kernel_sum = np.sum(np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=-1)
            else:
                kernel_sum = np.sum(np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=0)

            remaining_filter_indices = np.sort(np.argsort(kernel_sum)[m:])
            remaining_filter_indices_list.append(remaining_filter_indices)

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

        # elif isinstance(layer, keras.layers.Dense):
        #     raise NotImplementedError

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
        if not isinstance(submodel.layers[-2], keras.layers.Dense) or not isinstance(submodel.layers[-2],
                                                                                     keras.layers.Conv2D):
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

    return pruned_submodel


def restore_layer_names(layer, pruned_submodel):
    layer_config = layer.layers[-1].get_config()
    layer_config['name'] = 'pruned_' + layer_config['name']
    new_layer = type(layer).from_config(layer_config)
    if pruned_submodel.layers[-2].name.startswith('pruned_'):
        pass  # adjust connections
    raise NotImplementedError
