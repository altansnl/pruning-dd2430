import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import clone_model
from keras.utils.layer_utils import count_params
from tensorflow import keras
import time
import numpy as np

import pruning
import flops
from model import CVAE
import utils


# Deterministic run
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

# todo Show ratio
# todo m kac olmalı? strategy
# todo farklı datasetler
# todo dense layer
# todo gradient, batch norm techniques
# todo baselines

latent_dim = 2
train_size = 60000
batch_size = 32
test_size = 10000
epochs = 5
num_examples_to_generate = 16
optimizer = tf.keras.optimizers.Adam(1e-4)

num_pruning_iterations = 3
rewind_weights_epoch = 3  # False|epoch number - reverts weights to initial random initialization or specified epoch


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


initial_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Flatten(),
                tf.keras.layers.GlobalAveragePooling2D(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ],
            name="Encoder"
        )

initial_decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dense(256, activation="relu"),
                # tf.keras.layers.Dense(784),
                # tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                   filters=64, kernel_size=3, strides=2, padding='same',
                   activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                   filters=32, kernel_size=3, strides=2, padding='same',
                   activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                   filters=1, kernel_size=3, strides=1, padding='same'),
            ],
            name="Decoder"
        )

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

##############################################
scenarios = [1, 2, 3, 4]
scenario_labels = ["Original", "Only Encoder", "Only Decoder", "Both Encoder and Decoder"]

fig, ax = plt.subplots(2, 2, figsize=(11, 7))

for no, scenario in enumerate(scenarios):
    scenario_elbos_list = []
    total_flops_ratio_list = []
    mean_inference_time_ratio_list = []
    total_params_ratio_list = []

    cvae = CVAE(clone_model(initial_encoder), clone_model(initial_decoder), latent_dim)
    print(f"Scenario: {scenario_labels[no]}")

    for pruning_iteration in range(num_pruning_iterations):
        # cvae.encoder.summary()
        # cvae.decoder.summary()

        for epoch in range(1, epochs + 1):
            # train_loss = tf.keras.metrics.Mean()
            for train_x in train_dataset:
                cvae.train_step(train_x, optimizer)
                # train_loss(cvae.compute_loss(train_x))
            # train_elbo = -train_loss.result()

            test_loss = tf.keras.metrics.Mean()
            start_time = time.time()
            for test_x in test_dataset:
                test_loss(cvae.compute_loss(test_x))
            end_time = time.time()

            test_elbo = -test_loss.result()
            scenario_elbos_list.append(test_elbo.numpy())

            # print('Epoch: {}, Train set ELBO: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            #       .format(epoch, train_elbo, test_elbo, end_time - start_time))
            print('Epoch: {}, Test set ELBO: {}, Inference time: {}'
                  .format(epoch, test_elbo, end_time - start_time))

            if epoch == rewind_weights_epoch:
                print("rewind weights are saved!")
                rewind_model = CVAE(clone_model(cvae.encoder), clone_model(cvae.decoder), latent_dim)
                rewind_model.encoder.set_weights(cvae.encoder.get_weights())
                rewind_model.decoder.set_weights(cvae.decoder.get_weights())

        mean_inference_time, total_flops, total_params = utils.calculate_metrics(cvae, test_dataset)

        if pruning_iteration == 0:
            mean_original_inference_time = mean_inference_time

        mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time

        if pruning_iteration == 0:
            original_total_flops = total_flops

        total_flops_ratio = 100 * total_flops / original_total_flops

        if pruning_iteration == 0:
            original_total_params = total_params

        total_params_ratio = 100 * total_params / original_total_params

        mean_inference_time_ratio_list.append(mean_inference_time_ratio)
        total_flops_ratio_list.append(total_flops_ratio)
        total_params_ratio_list.append(total_params_ratio)

        m = 2
        # local/layer-wise cnn pruning
        cvae = pruning.structural_prune(cvae, rewind_model, m, scenario)
        print("pruned and rewinded!")

    mean_inference_time, total_flops, total_params = utils.calculate_metrics(cvae, test_dataset)

    mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time
    total_flops_ratio = 100 * total_flops / original_total_flops
    total_params_ratio = 100 * total_params / original_total_params

    mean_inference_time_ratio_list.append(mean_inference_time_ratio)
    total_flops_ratio_list.append(total_flops_ratio)
    total_params_ratio_list.append(total_params_ratio)

    ax[0, 0].plot(np.arange(1, len(scenario_elbos_list) + 1), scenario_elbos_list, label=scenario_labels[no])
    ax[0, 1].plot(np.arange(0, len(mean_inference_time_ratio_list)), mean_inference_time_ratio_list)
    ax[1, 0].plot(np.arange(0, len(total_flops_ratio_list)), total_flops_ratio_list)
    ax[1, 1].plot(np.arange(0, len(total_params_ratio_list)), total_params_ratio_list)

# ax[0, 0].legend()
ax[0, 0].set_xlabel("Epochs")
ax[0, 0].set_ylabel("NLL")

# ax[0, 1].legend()
ax[0, 1].set_xlabel("Pruning Iterations")
ax[0, 1].set_ylabel("Mean Inference Time %")

# ax[1, 0].legend()
ax[1, 0].set_xlabel("Pruning Iterations")
ax[1, 0].set_ylabel("FLOPs %")

# ax[1, 1].legend()
ax[1, 1].set_xlabel("Pruning Iterations")
ax[1, 1].set_ylabel("Params %")

# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)

fig.legend(loc=7)
fig.suptitle("VAE Pruning every 5 epoch. Rewind to 3rd epoch.")
fig.tight_layout()
fig.subplots_adjust(right=0.75)
fig.savefig("results.png")
fig.show()
