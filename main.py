import tensorflow as tf
from tensorflow import keras
import time
from cvae import CVAE
import numpy as np
import tensorflow_model_optimization as tfmot
from preprocess import preprocess_images
from numpy import linalg as LA

latent_dim = 2
train_size = 60000
batch_size = 32
test_size = 10000
epochs = 1
end_step = np.ceil(train_size / batch_size).astype(np.int32) * epochs
num_examples_to_generate = 16
optimizer = tf.keras.optimizers.Adam(1e-4)

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.9999, begin_step=0, end_step=3),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
}

if __name__ == "__main__":

    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim]
    )

    cvae = CVAE(latent_dim, pruning_params)
    cvae.compile(optimizer=optimizer)
    
    (train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

    

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(cvae.encoder)

    step_callback.on_train_begin() # run pruning callback
    for epoch in range(1, epochs+ 1):
        i = 0
        start_time = time.time()
        for train_x in train_dataset:
            step_callback.on_train_batch_begin(-1) # run pruning callback
            cvae.train_step(train_x, optimizer)
            i += 1
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(cvae.compute_loss(test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
        step_callback.on_epoch_end(batch=-1) # run pruning callback