import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import clone_model
from tensorflow import keras
import time
import numpy as np
import pruning
from model import CVAE
import utils

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

# HYPER-PARAMETERS FOR EXPERIMENTS
LATENT_DIM             = 8       # isn't this too low?
TRAIN_SIZE             = 60000
BATCH_SIZE             = 64
TEST_SIZE              = 10000
EPOCH_NORMAL_FIT       = 30       # number of epochs to be ran before the prunning cycles
NUM_PRUNING_CYCLES     = 5
EPOCH_PRUNING_CYCLE    = 5
REWIND_WEIGHTS_EPOCH   = 3       # reverts weights to initial random initialization or specified epoch
FINAL_PRUNE_PERCENTAGE = 0.9

SCENARIOS = [1, 2, 3, 4]
SCENARIO_LABELS = ["Original", "Only Encoder", "Only Decoder", "Both Encoder and Decoder"]

optimizer = tf.keras.optimizers.Adam(1e-4)

def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

if __name__ == "__main__":
    initial_encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    # No activation
                    tf.keras.layers.Dense(LATENT_DIM + LATENT_DIM),
                ],
                name="Encoder"
            )

    initial_decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(LATENT_DIM,)),
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
                    .shuffle(TRAIN_SIZE).batch(BATCH_SIZE))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(TEST_SIZE).batch(BATCH_SIZE))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    for no, scenario in enumerate(SCENARIOS):
        elbos_list = []
        elbos_epochs = []
        total_flops_ratio_list = []
        mean_inference_time_ratio_list = []
        total_params_ratio_list = []

        def train_epoch(epoch_number, test_first=False):
            test_loss = tf.keras.metrics.Mean()
            if test_first:
                for test_x in test_dataset:
                    test_loss(cvae.compute_loss(test_x))
                test_elbo = -test_loss.result()
                elbos_list.append(test_elbo.numpy())
                elbos_epochs.append(epoch_number-0.01)
                print(f'Epoch: {epoch_number}, Test set ELBO: {test_elbo}') 
            for train_x in train_dataset:
                cvae.train_step(train_x, optimizer)
            start_time = time.time()
            for test_x in test_dataset:
                test_loss(cvae.compute_loss(test_x))
            end_time = time.time()
            test_elbo = -test_loss.result()
            elbos_epochs.append(epoch_number+0.01)
            elbos_list.append(test_elbo.numpy())
            print(f'Epoch: {epoch_number}, Test set ELBO: {test_elbo}, Inference time: {end_time - start_time}')   

        cvae = CVAE(clone_model(initial_encoder), clone_model(initial_decoder), LATENT_DIM)
        print(f"Scenario: {SCENARIO_LABELS[no]}")

        e = 1
        for epoch in range(1, EPOCH_NORMAL_FIT+1):
            train_epoch(epoch_number=e)
            e += 1

        for pruning_iteration in range(NUM_PRUNING_CYCLES+1):
            for epoch in range(1, EPOCH_PRUNING_CYCLE + 1):
                first_epoch_in_prunning = epoch == 1
                train_epoch(epoch_number=e, test_first=False)
                e += 1

                if epoch == REWIND_WEIGHTS_EPOCH:
                    print("rewind weights are saved!")
                    rewind_model = CVAE(clone_model(cvae.encoder), clone_model(cvae.decoder), LATENT_DIM)
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

            if pruning_iteration != NUM_PRUNING_CYCLES:
                if pruning_iteration == 0:
                    left_to_prune_encoder = None
                    left_to_prune_decoder = None
                print("maps before pruning")
                print(left_to_prune_encoder, "\n", left_to_prune_decoder)
                cvae, left_to_prune_encoder, left_to_prune_decoder = pruning.structural_prune(
                    cvae, 
                    rewind_model, 
                    scenario, 
                    left_to_prune_encoder, 
                    left_to_prune_decoder, 
                    FINAL_PRUNE_PERCENTAGE, 
                    NUM_PRUNING_CYCLES-pruning_iteration
                )
                print("maps after pruning")
                print(left_to_prune_encoder, "\n", left_to_prune_decoder)
                print("pruned and rewinded!")

        mean_inference_time, total_flops, total_params = utils.calculate_metrics(cvae, test_dataset)
        mean_inference_time_ratio = 100 * mean_inference_time / mean_original_inference_time
        total_flops_ratio = 100 * total_flops / original_total_flops
        total_params_ratio = 100 * total_params / original_total_params
        mean_inference_time_ratio_list.append(mean_inference_time_ratio)
        total_flops_ratio_list.append(total_flops_ratio)
        total_params_ratio_list.append(total_params_ratio)
        ax[0, 0].plot(elbos_epochs, elbos_list, label=SCENARIO_LABELS[no])
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

    fig.legend(loc=7)
    total_epochs = EPOCH_NORMAL_FIT+(NUM_PRUNING_CYCLES+1)*EPOCH_PRUNING_CYCLE
    fig.suptitle(f"Pruning metrics (prune percetange:{FINAL_PRUNE_PERCENTAGE}, total epochs:{total_epochs}, prune cycles:{NUM_PRUNING_CYCLES})")
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    fig.savefig(f"outputs/{int(FINAL_PRUNE_PERCENTAGE*100)}-{NUM_PRUNING_CYCLES}.png")
    fig.show()
