import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

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


class CVAE(tf.keras.Model):
    """Fully dense variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
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
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(784),
                tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
                # tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                # tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                # tf.keras.layers.Conv2DTranspose(
                #    filters=64, kernel_size=3, strides=2, padding='same',
                #    activation='relu'),
                # tf.keras.layers.Conv2DTranspose(
                #    filters=32, kernel_size=3, strides=2, padding='same',
                #    activation='relu'),
                # No activation
                # tf.keras.layers.Conv2DTranspose(
                #    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    # @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    # @tf.function
    def train_step(self, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))


random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim]
)

cvae = CVAE(latent_dim)

(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

##############################################

for pruning_iteration in range(num_pruning_iterations):
    pruned_encoder = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28, 1))])

    cvae.encoder.summary()
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        for train_x in train_dataset:
            cvae.train_step(train_x, optimizer)

        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(cvae.compute_loss(test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))

        if epoch == rewind_weights_epoch:
            rewind_model = cvae

    remaining_filter_indices_list = []
    m = 2
    # local/layer-wise cnn pruning
    for no, layer in enumerate(cvae.encoder.layers[:-1]):  # Exclude last layer
        if isinstance(layer, keras.layers.Dense):
            raise NotImplementedError

        elif isinstance(layer, keras.layers.Conv2D):
            conv_weights, conv_biases = layer.get_weights()
            kernel_sum = np.sum(np.sum(np.sum(abs(conv_weights), axis=0), axis=0), axis=0)
            remaining_filter_indices = np.sort(np.argsort(kernel_sum)[m:])
            remaining_filter_indices_list.append(remaining_filter_indices)

            # add pruned layer to new model
            new_layer = tf.keras.layers.Conv2D(
                filters=len(remaining_filter_indices),
                kernel_size=conv_weights.shape[:2],
                strides=(2, 2),
                activation='relu'
            )
            pruned_encoder.add(new_layer)

            # Get rewind weights
            rewind_weights, rewind_biases = rewind_model.encoder.layers[no].get_weights()

            if isinstance(cvae.encoder.layers[no - 1], keras.layers.Conv2D):
                previous_layer_remaining_filter_indices = remaining_filter_indices_list[-2]
                updated_rewind_weights = rewind_weights[:, :, previous_layer_remaining_filter_indices, :]

                pruned_encoder.layers[-1].set_weights([
                    updated_rewind_weights[:, :, :, remaining_filter_indices],
                    rewind_biases[remaining_filter_indices]
                ])

            else:
                pruned_encoder.layers[-1].set_weights([
                    rewind_weights[:, :, :, remaining_filter_indices],
                    rewind_biases[remaining_filter_indices]
                ])

        elif isinstance(layer, keras.layers.Flatten):
            raise NotImplementedError  # requires determination of the removed connection for the next dense layer.

        elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
            pruned_encoder.add(tf.keras.layers.GlobalAveragePooling2D())

        else:
            raise Exception("Layer type not implemented!")

    # Add last layer
    new_output_layer = tf.keras.layers.Dense(latent_dim + latent_dim)
    pruned_encoder.add(new_output_layer)
    # Get rewind weights
    output_rewind_weights, output_rewind_biases = rewind_model.encoder.layers[-1].get_weights()

    previous_layer_remaining_filter_indices = remaining_filter_indices_list[-1]
    pruned_encoder.layers[-1].set_weights([
        output_rewind_weights[previous_layer_remaining_filter_indices, :],
        output_rewind_biases
    ])

    cvae.encoder = pruned_encoder
    print("Encoder pruned and rewinded!")
