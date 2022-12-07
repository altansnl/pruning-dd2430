import time

import numpy as np
from keras.utils.layer_utils import count_params
import tensorflow as tf
import flops


def calculate_metrics(cvae, test_dataset):
    # Time
    inference_time = []
    for i in range(5):
        inference_metric = tf.keras.metrics.Mean()
        start_time = time.time()
        for test_x in test_dataset:
            inference_metric(cvae.compute_loss(test_x))
        end_time = time.time()
        inference_time.append(end_time - start_time)

    mean_inference_time = np.mean(np.array(inference_time))

    # FLOPs
    sample = next(iter(test_dataset))
    enc_sample = tf.constant(np.expand_dims(sample[0], axis=0))
    encoder_flops = flops.get_flops(cvae.encoder, [enc_sample])
    dec_sample = tf.constant(np.expand_dims(cvae.encoder(sample)[0, :cvae.latent_dim], axis=0))
    decoder_flops = flops.get_flops(cvae.decoder, [dec_sample])
    total_flops = encoder_flops + decoder_flops

    # Parameters
    enc_trainable_params = sum(count_params(layer) for layer in cvae.encoder.trainable_weights)
    enc_non_trainable_params = sum(count_params(layer) for layer in cvae.encoder.non_trainable_weights)
    dec_trainable_params = sum(count_params(layer) for layer in cvae.decoder.trainable_weights)
    dec_non_trainable_params = sum(count_params(layer) for layer in cvae.decoder.non_trainable_weights)

    total_params = enc_trainable_params + enc_non_trainable_params + dec_trainable_params + dec_non_trainable_params

    return mean_inference_time, total_flops, total_params
