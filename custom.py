import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import clone_model

from tensorflow import keras
import time
import numpy as np
import pruning
from model import CVAE

import utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

# HYPER-PARAMETERS FOR EXPERIMENTS
LATENT_DIM = 8       # isn't this too low?
BUFFER_SIZE = 60000
ACT_TRAIN_SIZE = 60000
BATCH_SIZE = 64
TEST_SIZE = 10000
ACT_TEST_SIZE = 10000
EPOCH_NORMAL_FIT = 10   # 30       # number of epochs to be ran before the prunning cycles
NUM_PRUNING_CYCLES = 7  # 5
EPOCH_PRUNING_CYCLE = 5  # 5
# reverts weights to initial random initialization or specified epoch
REWIND_WEIGHTS_EPOCH = 3  # 3
FINAL_PRUNE_PERCENTAGE = 0.5

SCENARIOS = [1, 2, 3, 4]
SCENARIO_LABELS = ["Only Encoder",
                   "Only Decoder", "Both Encoder and Decoder", "Original"]
optimizer = tf.keras.optimizers.Adam(1e-4)


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


if __name__ == "__main__":
    initial_encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            # No activation
            tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM),
        ],
        name="Encoder"
    )

    # initial_decoder = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
    #         tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
    #         tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=64, kernel_size=3, strides=2, padding='same',
    #             activation='relu'),
    #         # No activation
    #         tf.keras.layers.Conv2DTranspose(
    #             filters=1, kernel_size=3, strides=1, padding='same'),
    #     ],
    #     name="Decoder"
    # )
    initial_decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
            tf.keras.layers.Reshape(target_shape=(1, 1, LATENT_DIM,)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=1, padding='valid',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='valid',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=2, padding='same'),
        ],
        name="Decoder"
    )

    # (train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
    (train_images, _), (test_images, _) = keras.datasets.fashion_mnist.load_data()

    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    DEBUG_TRAIN_SET = 1
    if DEBUG_TRAIN_SET:
        train_images = train_images[:ACT_TRAIN_SIZE, :, :, :]
        test_images = test_images[:ACT_TEST_SIZE, :, :, :]
        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                         .shuffle(BUFFER_SIZE).batch(BATCH_SIZE))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(TEST_SIZE).batch(BATCH_SIZE))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    plt.subplots_adjust(left=0.4, wspace=0.2, hspace=0.2)
    prune_percentages = [0.2, 0.4, 0.6, 0.8]
    for prune_perc in prune_percentages:
        FINAL_PRUNE_PERCENTAGE = prune_perc
        SCENARIOS = [2]
        assert len(SCENARIOS) == 1
        print("[LOG] Pruning:", FINAL_PRUNE_PERCENTAGE,
              "on", SCENARIO_LABELS[SCENARIOS[0] - 1])
        for no, scenario in enumerate(SCENARIOS):
            elbos_list = []
            elbos_epochs = []
            total_flops_ratio_list = []
            mean_inference_time_ratio_list = []
            total_params_ratio_list = []
            elbos_list_pruning_iter = []  # used to store the elbos after each pruning iteration

            def train_epoch(epoch_number, test_first=False):
                test_loss = tf.keras.metrics.Mean()
                if test_first:
                    for test_x in test_dataset:
                        test_loss(cvae.compute_loss(test_x))
                    test_elbo = -test_loss.result()
                    elbos_list.append(test_elbo.numpy())
                    elbos_epochs.append(epoch_number - 0.01)
                    print(f'Epoch: {epoch_number}, Test set ELBO: {test_elbo}')
                for train_x in train_dataset:
                    cvae.train_step(train_x, optimizer)
                start_time = time.time()
                for test_x in test_dataset:
                    test_loss(cvae.compute_loss(test_x))
                end_time = time.time()
                test_elbo = -test_loss.result()
                elbos_epochs.append(epoch_number + 0.01)
                elbos_list.append(test_elbo.numpy())
                print(
                    f'Epoch: {epoch_number}, Test set ELBO: {test_elbo}, Inference time: {end_time - start_time}')

            cvae = CVAE(clone_model(initial_encoder),
                        clone_model(initial_decoder), LATENT_DIM)
            print(f"Scenario: {SCENARIO_LABELS[scenario-1]}")

            e = 1
            for epoch in range(1, EPOCH_NORMAL_FIT + 1):
                train_epoch(epoch_number=e)
                e += 1

            for pruning_iteration in range(NUM_PRUNING_CYCLES + 1):
                for epoch in range(1, EPOCH_PRUNING_CYCLE + 1):
                    first_epoch_in_prunning = epoch == 1
                    train_epoch(epoch_number=e,
                                test_first=False)
                    e += 1

                    if epoch == REWIND_WEIGHTS_EPOCH:
                        print("rewind weights are saved!")
                        rewind_model = CVAE(clone_model(
                            cvae.encoder), clone_model(cvae.decoder), LATENT_DIM)
                        rewind_model.encoder.set_weights(
                            cvae.encoder.get_weights())
                        rewind_model.decoder.set_weights(
                            cvae.decoder.get_weights())

                mean_inference_time, total_flops, total_params = utils.save_and_calculate_metrics(
                    cvae, test_dataset, scenario)

                if pruning_iteration == 0:
                    mean_original_inference_time = mean_inference_time

                mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time

                if pruning_iteration == 0:
                    original_total_flops = total_flops

                total_flops_ratio = 100 * total_flops / original_total_flops

                if pruning_iteration == 0:
                    original_total_params = total_params

                total_params_ratio = 100 * total_params / original_total_params

                mean_inference_time_ratio_list.append(
                    mean_inference_time_ratio)
                total_flops_ratio_list.append(total_flops_ratio)
                total_params_ratio_list.append(total_params_ratio)
                elbos_list_pruning_iter.append(elbos_list[-1])

                if pruning_iteration != NUM_PRUNING_CYCLES:
                    if pruning_iteration == 0:
                        left_to_prune_encoder = None
                        left_to_prune_decoder = None
                    # print("maps before pruning")
                    # print(left_to_prune_encoder, "\n", left_to_prune_decoder)
                    cvae, left_to_prune_encoder, left_to_prune_decoder = pruning.structural_prune(
                        cvae,
                        rewind_model,
                        scenario,
                        left_to_prune_encoder,
                        left_to_prune_decoder,
                        FINAL_PRUNE_PERCENTAGE,
                        NUM_PRUNING_CYCLES - pruning_iteration
                    )
                    # print("maps after pruning")
                    # print(left_to_prune_encoder, "\n", left_to_prune_decoder)
                    # print("pruned and rewinded!")

            elbos_list = [x * -1 for x in elbos_list]
            elbos_list_pruning_iter = [x * -1 for x in elbos_list_pruning_iter]
            # TODO: COMMENT IN FOR NON-PERCENTAGE EXP
            # ax[0, 0].plot(elbos_epochs, elbos_list,
            #               label=SCENARIO_LABELS[scenario - 1])
            # TODO: BELOW LINE ONLY FOR PERCENTAGE EXP
            ax[0, 0].plot(elbos_epochs, elbos_list,
                          label=str(prune_perc))
            ax[0, 0].set_xlabel("Epochs")
            ax[0, 0].set_ylabel("NLL")
            print("Scenario", scenario)
            if scenario != 4:
                ax[0, 1].plot(total_params_ratio_list,
                              mean_inference_time_ratio_list)
                ax[0, 1].set_xlabel("Params %")
                ax[0, 1].set_ylabel("Mean Inference Time %")

                ax[1, 0].plot(
                    total_flops_ratio_list, elbos_list_pruning_iter)
                ax[1, 0].set_xlabel("FLOPs %")
                ax[1, 0].set_ylabel("NLL")
                ax[1, 1].plot(
                    total_params_ratio_list, elbos_list_pruning_iter)
                ax[1, 1].set_xlabel("Params %")
                ax[1, 1].set_ylabel("NLL")
    ax[0, 1].invert_xaxis()

    ax[1, 0].invert_xaxis()

    ax[1, 1].invert_xaxis()

    fig.legend(loc=7)
    fig.suptitle(
        f"Pruning {SCENARIO_LABELS[SCENARIOS[0]-1]} (prune percetange:{FINAL_PRUNE_PERCENTAGE}, total epoch:{EPOCH_NORMAL_FIT+NUM_PRUNING_CYCLES*EPOCH_PRUNING_CYCLE}, prune cycles:{NUM_PRUNING_CYCLES})")
    # fig.suptitle(
    #     f"Pruning metrics (prune percetange:{FINAL_PRUNE_PERCENTAGE}, total epoch:{EPOCH_NORMAL_FIT+NUM_PRUNING_CYCLES*EPOCH_PRUNING_CYCLE}, prune cycles:{NUM_PRUNING_CYCLES})")
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig("results" + str(SCENARIO_LABELS[SCENARIOS[0] - 1]) + ".png")
    fig.show()
