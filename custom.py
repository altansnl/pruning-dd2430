import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import clone_model
from tensorflow import keras
import time
import numpy as np

import pruning
from model import CVAE

# todo kazanc ne kadar metrikleri koy.
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
# False|epoch number - reverts weights to initial random initialization or specified epoch
rewind_weights_epoch = 3


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
        tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
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
scenario_labels = ["Original", "Only Encoder",
                   "Only Decoder", "Both Encoder and Decoder"]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

for no, scenario in enumerate(scenarios):
    scenario_elbos = []
    inference_time = []
    cvae = CVAE(clone_model(initial_encoder),
                clone_model(initial_decoder), latent_dim)
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
            scenario_elbos.append(test_elbo.numpy())

            # print('Epoch: {}, Train set ELBO: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            #       .format(epoch, train_elbo, test_elbo, end_time - start_time))
            print('Epoch: {}, Test set ELBO: {}, Inference time: {}'
                  .format(epoch, test_elbo, end_time - start_time))

            if epoch == rewind_weights_epoch:
                print("rewind weights are saved!")
                rewind_model = CVAE(clone_model(cvae.encoder),
                                    clone_model(cvae.decoder), latent_dim)
                rewind_model.encoder.set_weights(cvae.encoder.get_weights())
                rewind_model.decoder.set_weights(cvae.decoder.get_weights())

        inference_metric = tf.keras.metrics.Mean()
        start_time = time.time()
        for test_x in test_dataset:
            inference_metric(cvae.compute_loss(test_x))
        end_time = time.time()

        inference_time.append(end_time - start_time)

        m = 2
        # local/layer-wise cnn pruning
        cvae = pruning.structural_prune(cvae, rewind_model, m, scenario)
        print("pruned and rewinded!")

    inference_metric = tf.keras.metrics.Mean()
    start_time = time.time()
    for test_x in test_dataset:
        inference_metric(cvae.compute_loss(test_x))
    end_time = time.time()

    inference_time.append(end_time - start_time)

    ax[1].plot(np.arange(0, len(inference_time)),
               inference_time, label=scenario_labels[no])
    ax[0].plot(np.arange(1, len(scenario_elbos) + 1),
               scenario_elbos, label=scenario_labels[no])

ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("NLL")

ax[1].legend()
ax[1].set_xlabel("Pruning Iterations")
ax[1].set_ylabel("Inference Time (s)")

fig.suptitle("VAE Pruning every 5 epoch. Rewind to 3rd epoch.")
fig.savefig("arch_exp.png")
fig.show()
