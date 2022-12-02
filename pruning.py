import tensorflow as tf
from keras.saving.save import load_model
from tensorflow import keras
import numpy as np
import tensorflow_addons as tfa

from model import CVAE

LAYER = 0
VALUE = 1
INDEX = 2


def structural_prune(cvae: CVAE, rewind_cvae: CVAE, m, scenario):
    if scenario == 1:
        pruned_cvae = cvae

    elif scenario == 2:
        pruned_encoder = _global_structural_prune_submodel(
            cvae.encoder, rewind_cvae.encoder, m)
        pruned_encoder.save('saved_models/pruned_encoder.h5')
        pruned_encoder = load_model('saved_models/pruned_encoder.h5')

        pruned_decoder = cvae.decoder
        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)

    elif scenario == 3:
        pruned_encoder = cvae.encoder

        pruned_decoder = _global_structural_prune_submodel(
            cvae.decoder, rewind_cvae.decoder, m)
        pruned_decoder.save('saved_models/pruned_decoder.h5')
        pruned_decoder = load_model('saved_models/pruned_decoder.h5')

        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)

    elif scenario == 4:
        pruned_encoder = _global_structural_prune_submodel(
            cvae.encoder, rewind_cvae.encoder, m)
        pruned_encoder.save('saved_models/pruned_encoder.h5')
        pruned_encoder = load_model('saved_models/pruned_encoder.h5')
        print("[LOG] Pruning Encoder done.")
        pruned_decoder = _global_structural_prune_submodel(
            cvae.decoder, rewind_cvae.decoder, m)
        pruned_decoder.save('saved_models/pruned_decoder.h5')
        pruned_decoder = load_model('saved_models/pruned_decoder.h5')
        print("[LOG] Pruning Decoder done.")
        pruned_cvae = CVAE(pruned_encoder, pruned_decoder, cvae.latent_dim)
    else:
        raise ValueError("Not applicable Scenario.")

    return pruned_cvae


def _global_structural_prune_submodel(submodel: tf.keras.Sequential, rewind_submodel: tf.keras.Sequential, m):
    # Add input layer
    input_layer = tf.keras.layers.InputLayer(
        input_shape=submodel.layers[0].input_shape[1:])
    pruned_submodel = tf.keras.Sequential([input_layer])

    # layer_tupels = (layer, original_index, original_value), used for global pruning
    layer_tupels = np.array([]).reshape(0, 3)
    # Exclude last layer

    for no, layer in enumerate(submodel.layers[:-1]):
        if isinstance(layer, keras.layers.Conv2D):

            conv_weights, conv_biases = layer.get_weights()
            if isinstance(layer, keras.layers.Conv2DTranspose):
                kernel_sum = np.sum(
                    np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=-1)
            else:
                kernel_sum = np.sum(
                    np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=0)

            normed_kernel_sum = kernel_sum / np.linalg.norm(kernel_sum)
            layer_tupel = np.column_stack(
                (np.full((kernel_sum.shape), no, dtype=int), normed_kernel_sum, np.arange(0, kernel_sum.shape[0])))
            layer_tupels = np.concatenate((layer_tupels, layer_tupel))

            # create tupels (layer,original_index,original_value)
    sorted_layer_tuples = layer_tupels[layer_tupels[:, 1].argsort()]
    # print('Sorted Layers:\n', sorted_layer_tuples, '\n')
    filters_to_prune = sorted_layer_tuples[:m]
    filters_to_prune = filters_to_prune[filters_to_prune[:, 2].argsort()]
    remaining_tuples = sorted_layer_tuples[m:]
    # print('Filters to prune:\n', filters_to_prune, '\n')
    print('\n Filters to prune:')
    for filter in filters_to_prune:
        print('-',
              int(filter[LAYER]), '/', int(filter[INDEX]), '/', round(filter[VALUE], 3), '(Layer/Index/Value')
    remaining_filter_indices_list = []
    for layer_no, current_layer in enumerate(submodel.layers[:-1]):
        # if layer_no in filters_to_prune[:, LAYER]:
        if isinstance(current_layer, keras.layers.Conv2D):

            remaining_filter_indices = remaining_tuples[np.where(
                remaining_tuples[:, LAYER] == layer_no)][:, INDEX].astype(int)
            print('\nRemaining filters for layer', layer_no,
                  ':', len(remaining_filter_indices))
            remaining_filter_indices_list.append(remaining_filter_indices)

            layer_config = current_layer.get_config()
            layer_config['filters'] = len(remaining_filter_indices)
            new_layer = type(current_layer).from_config(layer_config)
            pruned_submodel.add(new_layer)

            # Get rewind weights
            rewind_weights, rewind_biases = rewind_submodel.layers[layer_no].get_weights(
            )

            if isinstance(submodel.layers[layer_no - 1], keras.layers.Conv2D):
                previous_layer_remaining_filter_indices = remaining_filter_indices_list[-2]

                if isinstance(submodel.layers[layer_no - 1], keras.layers.Conv2DTranspose):
                    updated_rewind_weights = rewind_weights[:, :,
                                                            :, previous_layer_remaining_filter_indices]
                else:
                    updated_rewind_weights = rewind_weights[:, :,
                                                            previous_layer_remaining_filter_indices, :]

                if isinstance(current_layer, keras.layers.Conv2DTranspose):
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :,
                                               remaining_filter_indices, :],
                        rewind_biases[remaining_filter_indices]
                    ])

                else:
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :, :,
                                               remaining_filter_indices],
                        rewind_biases[remaining_filter_indices]
                    ])

            else:
                if isinstance(current_layer, keras.layers.Conv2DTranspose):
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
            config = rewind_submodel.layers[layer_no].get_config()
            weights = rewind_submodel.layers[layer_no].get_weights()
            # print("Layer", layer_no, ", config:\n", config)
            cloned_layer = type(
                rewind_submodel.layers[layer_no]).from_config(config)

            pruned_submodel.add(cloned_layer)

            pruned_submodel.layers[-1].set_weights(weights)

    # END TODO
    # Add last layer
    if isinstance(submodel.layers[-1], keras.layers.Conv2D):
        output_layer_config = submodel.layers[-1].get_config()

        output_layer = type(
            submodel.layers[-1]).from_config(output_layer_config)

        pruned_submodel.add(output_layer)

        # Get rewind weights
        rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights(
        )

        if isinstance(submodel.layers[-2], keras.layers.Conv2D):
            previous_layer_remaining_filter_indices = remaining_filter_indices_list[-1]
            if isinstance(submodel.layers[-2], keras.layers.Conv2DTranspose):
                updated_rewind_weights = rewind_weights[:, :,
                                                        :, previous_layer_remaining_filter_indices]
            else:
                updated_rewind_weights = rewind_weights[:, :,
                                                        previous_layer_remaining_filter_indices, :]

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
                output_layer = type(
                    rewind_submodel.layers[-1]).from_config(output_layer_config)
                pruned_submodel.add(output_layer)

                rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights(
                )
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

        output_layer = type(
            rewind_submodel.layers[-1]).from_config(output_layer_config)

        pruned_submodel.add(output_layer)

        pruned_submodel.layers[-1].set_weights(weights)

    return pruned_submodel


def _structural_prune_submodel(submodel: tf.keras.Sequential, rewind_submodel: tf.keras.Sequential, m):
    # Add input layer
    input_layer = tf.keras.layers.InputLayer(
        input_shape=submodel.layers[0].input_shape[1:])
    pruned_submodel = tf.keras.Sequential([input_layer])

    remaining_filter_indices_list = []
    for no, layer in enumerate(submodel.layers[:-1]):  # Exclude last layer
        if isinstance(layer, keras.layers.Conv2D):
            conv_weights, conv_biases = layer.get_weights()
            # print(conv_weights.shape)
            if isinstance(layer, keras.layers.Conv2DTranspose):
                kernel_sum = np.sum(
                    np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=-1)
            else:
                kernel_sum = np.sum(
                    np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=0)

            remaining_filter_indices = np.sort(np.argsort(kernel_sum)[m:])
            print(np.max(remaining_filter_indices))
            remaining_filter_indices_list.append(remaining_filter_indices)

            layer_config = layer.get_config()
            layer_config['filters'] = len(remaining_filter_indices)

            new_layer = type(layer).from_config(layer_config)
            print(new_layer)
            print(layer_config)
            pruned_submodel.add(new_layer)

            # Get rewind weights
            rewind_weights, rewind_biases = rewind_submodel.layers[no].get_weights(
            )

            if isinstance(submodel.layers[no - 1], keras.layers.Conv2D):
                previous_layer_remaining_filter_indices = remaining_filter_indices_list[-2]

                if isinstance(submodel.layers[no - 1], keras.layers.Conv2DTranspose):
                    updated_rewind_weights = rewind_weights[:, :,
                                                            :, previous_layer_remaining_filter_indices]
                else:
                    updated_rewind_weights = rewind_weights[:, :,
                                                            previous_layer_remaining_filter_indices, :]

                if isinstance(layer, keras.layers.Conv2DTranspose):
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :,
                                               remaining_filter_indices, :],
                        rewind_biases[remaining_filter_indices]
                    ])

                else:
                    pruned_submodel.layers[-1].set_weights([
                        updated_rewind_weights[:, :, :,
                                               remaining_filter_indices],
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

        output_layer = type(
            submodel.layers[-1]).from_config(output_layer_config)

        pruned_submodel.add(output_layer)

        # Get rewind weights
        rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights(
        )

        if isinstance(submodel.layers[-2], keras.layers.Conv2D):
            previous_layer_remaining_filter_indices = remaining_filter_indices_list[-1]
            if isinstance(submodel.layers[-2], keras.layers.Conv2DTranspose):
                updated_rewind_weights = rewind_weights[:, :,
                                                        :, previous_layer_remaining_filter_indices]
            else:
                updated_rewind_weights = rewind_weights[:, :,
                                                        previous_layer_remaining_filter_indices, :]

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
                output_layer = type(
                    rewind_submodel.layers[-1]).from_config(output_layer_config)
                pruned_submodel.add(output_layer)

                rewind_weights, rewind_biases = rewind_submodel.layers[-1].get_weights(
                )
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

        output_layer = type(
            rewind_submodel.layers[-1]).from_config(output_layer_config)

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
