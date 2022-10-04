import tensorflow as tf
from tensorflow import keras
import time
from cvae import CVAE
import numpy as np
import tensorflow_model_optimization as tfmot
from preprocess import preprocess_images
from pruning_helpers import apply_pruning_to_dense

latent_dim = 2
train_size = 60000
batch_size = 32
test_size = 10000
epochs = 2
end_step = np.ceil(train_size / batch_size).astype(np.int32) * epochs
num_examples_to_generate = 16
optimizer = tf.keras.optimizers.Adam(1e-4)

if __name__ == "__main__":

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim]
    )

    cvae = CVAE(latent_dim)
    cvae.compile(optimizer=optimizer)
    
    (train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

    cvae.encoder.summary()
    cvae.decoder.summary()

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


    pruned_encoder = tf.keras.models.clone_model(
        cvae.encoder,
        clone_function=apply_pruning_to_dense,
    )
    
    pruned_encoder = tfmot.sparsity.keras.strip_pruning(pruned_encoder)

    print(np.sum([np.count_nonzero(x) for x in pruned_encoder.get_weights()]))

    tfmot.sparsity.keras.strip_pruning(pruned_encoder).summary()
    

    pruned_decoder = tf.keras.models.clone_model(
        cvae.decoder,
        clone_function=apply_pruning_to_dense,
    )

    pruned_decoder = tfmot.sparsity.keras.strip_pruning(pruned_decoder)

    print(np.sum([np.count_nonzero(x) for x in pruned_decoder.get_weights()]))

    tfmot.sparsity.keras.strip_pruning(pruned_decoder).summary()

