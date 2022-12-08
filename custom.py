import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import clone_model
from tensorflow import keras

import pruning
from model import CVAE
import utils

# todo different datasets
# todo dense layer
# todo gradient, batch norm techniques
# todo baselines

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()

# HYPER-PARAMETERS FOR EXPERIMENTS
LATENT_DIM = 32
TRAIN_SIZE = 60000
BATCH_SIZE = 32
TEST_SIZE = 10000
NUM_PRUNING_CYCLES = 5
EPOCH_PRUNING_CYCLE = 10
REWIND_WEIGHTS_EPOCH = 6  # reverts weights to initial random initialization or specified epoch
FINAL_PRUNE_PERCENTAGE = 0.5

SCENARIOS = [1, 2, 3, 4]
SCENARIO_LABELS = ["Original", "Only Encoder", "Only Decoder", "Both Encoder and Decoder"]
# SCENARIOS = [1]
# SCENARIO_LABELS = ["Original"]

optimizer = tf.keras.optimizers.Adam(1e-4)


def generate_images(model, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig2, ax2 = plt.subplots(4, figsize=(4, 4))

    for i in range(predictions.shape[0]):
        ax2[i].imshow(predictions[i, :, :, 0], cmap='gray')
        ax2[i].axis('off')

    fig2.show()


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


initial_encoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape(target_shape=(2304,)),
        tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM),
    ],
    name="Encoder"
)

initial_decoder = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
        tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
        tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=2, padding='same',
            activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=1, padding='same'),
    ],
    name="Decoder"
)

# initial_encoder = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
#         tf.keras.layers.Conv2D(
#             filters=32, kernel_size=3, strides=2, activation='relu', padding="same"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(
#             filters=64, kernel_size=3, strides=2, activation='relu', padding="same"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(
#             filters=128, kernel_size=3, strides=2, activation='relu', padding="valid"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(
#             filters=LATENT_DIM + LATENT_DIM, kernel_size=3, strides=1, activation='relu', padding="valid"),
#         tf.keras.layers.Reshape(target_shape=(LATENT_DIM + LATENT_DIM,))
#     ],
#     name="Encoder"
# )
#
# initial_decoder = tf.keras.Sequential(
#     [
#         tf.keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
#         tf.keras.layers.Reshape(target_shape=(1, 1, LATENT_DIM,)),
#         tf.keras.layers.Conv2DTranspose(
#             filters=128, kernel_size=3, strides=1, padding='valid',
#             activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2DTranspose(
#             filters=64, kernel_size=3, strides=2, padding='valid',
#             activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2DTranspose(
#             filters=32, kernel_size=3, strides=2, padding='same',
#             activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2DTranspose(
#             filters=1, kernel_size=3, strides=2, padding='same'),
#     ],
#     name="Decoder"
# )

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_dataset_raw = tf.data.Dataset.from_tensor_slices(train_images)
test_dataset_raw = tf.data.Dataset.from_tensor_slices(test_images)

for test_batch in test_dataset_raw.batch(BATCH_SIZE).take(1):
    test_sample = test_batch[0:4, :, :, :]

##############################################

fig, ax = plt.subplots(2, 2, figsize=(11, 7))

for no, scenario in enumerate(SCENARIOS):
    scenario_elbos_list = []
    total_flops_ratio_list = []
    mean_inference_time_ratio_list = []
    total_params_ratio_list = []

    # Create a new base model for each scenario.
    cvae = CVAE(clone_model(initial_encoder), clone_model(initial_decoder), LATENT_DIM)
    print(f"Scenario: {SCENARIO_LABELS[no]}")

    for pruning_iteration in range(NUM_PRUNING_CYCLES):
        # cvae.encoder.summary()
        # cvae.decoder.summary()

        for epoch in range(1, EPOCH_PRUNING_CYCLE + 1):
            train_dataset = train_dataset_raw.shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
            test_dataset = test_dataset_raw.shuffle(TEST_SIZE).batch(BATCH_SIZE)

            start_time = time.time()
            for train_x in train_dataset:
                cvae.train_step(train_x, optimizer)
            end_time = time.time()

            test_loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                test_loss(cvae.compute_loss(test_x))

            test_elbo = test_loss.result()
            scenario_elbos_list.append(test_elbo.numpy())

            print('Epoch: {}, Test set ELBO: {}, Training time: {}'
                  .format(epoch, test_elbo, end_time - start_time))

            if epoch == REWIND_WEIGHTS_EPOCH:
                print("Rewind weights are saved!")
                rewind_model = CVAE(clone_model(cvae.encoder), clone_model(cvae.decoder), LATENT_DIM)
                rewind_model.encoder.set_weights(cvae.encoder.get_weights())
                rewind_model.decoder.set_weights(cvae.decoder.get_weights())

        if pruning_iteration == 0:
            mean_inference_time, total_flops, total_params = utils.save_and_calculate_metrics(cvae, test_dataset,
                                                                                              scenario)
            mean_original_inference_time = mean_inference_time
            original_total_flops = total_flops
            original_total_params = total_params

            mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time
            total_flops_ratio = 100 * total_flops / original_total_flops
            total_params_ratio = 100 * total_params / original_total_params

            mean_inference_time_ratio_list.append(mean_inference_time_ratio)
            total_flops_ratio_list.append(total_flops_ratio)
            total_params_ratio_list.append(total_params_ratio)

        if pruning_iteration == 0:
            left_to_prune_encoder = None
            left_to_prune_decoder = None

        print("Maps before pruning")
        print(left_to_prune_encoder, "\n", left_to_prune_decoder)

        cvae, left_to_prune_encoder, left_to_prune_decoder = pruning.structural_prune(
            cvae,
            rewind_model,
            scenario,
            left_to_prune_encoder,
            left_to_prune_decoder,
            FINAL_PRUNE_PERCENTAGE,
            NUM_PRUNING_CYCLES - pruning_iteration
        )

        print("Maps after pruning")
        print(left_to_prune_encoder, "\n", left_to_prune_decoder)
        print("Pruned and Rewinded!")

        mean_inference_time, total_flops, total_params = utils.save_and_calculate_metrics(cvae, test_dataset,
                                                                                          scenario)

        mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time
        total_flops_ratio = 100 * total_flops / original_total_flops
        total_params_ratio = 100 * total_params / original_total_params

        mean_inference_time_ratio_list.append(mean_inference_time_ratio)
        total_flops_ratio_list.append(total_flops_ratio)
        total_params_ratio_list.append(total_params_ratio)

    ax[0, 0].plot(np.arange(1, len(scenario_elbos_list) + 1), scenario_elbos_list, label=SCENARIO_LABELS[no])
    ax[0, 1].plot(np.arange(0, len(mean_inference_time_ratio_list)), mean_inference_time_ratio_list)
    ax[1, 0].plot(np.arange(0, len(total_flops_ratio_list)), total_flops_ratio_list)
    ax[1, 1].plot(np.arange(0, len(total_params_ratio_list)), total_params_ratio_list)

    generate_images(cvae, test_sample)


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

date_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
results_dir = "results/" + date_string + "/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

total_epochs = EPOCH_PRUNING_CYCLE * NUM_PRUNING_CYCLES

fig.legend(loc=7)
fig.suptitle(
    f"Pruning metrics (prune percetange:{FINAL_PRUNE_PERCENTAGE}, total epochs:{total_epochs}, prune cycles:{NUM_PRUNING_CYCLES})")
fig.tight_layout()
fig.subplots_adjust(right=0.75)
fig.savefig(results_dir + f"pp_{int(FINAL_PRUNE_PERCENTAGE * 100)}.png")
fig.show()
