import time

import numpy as np
import tensorflow as tf
from keras.saving.save import load_model
from keras.utils.layer_utils import count_params
import flops
from model import CVAE


def save_models(encoder, decoder, scenario):
    encoder.save(f'saved_models/encoder-scenario-{scenario}.h5')
    decoder.save(f'saved_models/decoder-scenario-{scenario}.h5')


def reset_and_load_models(latent_dim, scenario):
    tf.keras.backend.clear_session()
    test_encoder = load_model(f'saved_models/encoder-scenario-{scenario}.h5', compile=False)
    test_decoder = load_model(f'saved_models/decoder-scenario-{scenario}.h5', compile=False)
    test_cvae = CVAE(test_encoder, test_decoder, latent_dim)

    return test_cvae


def calculate_time(test_dataset, latent_dim, scenario):
    # Reset graph before testing
    test_cvae = reset_and_load_models(latent_dim, scenario)
    decoder_inputs = test_dataset.map(lambda x: test_cvae.encoder(x, training=False)[..., :latent_dim])

    inference_time = []
    for i in range(1):  # 6
        start_time = time.time()
        test_cvae.encoder.predict(test_dataset)
        test_cvae.decoder.predict(decoder_inputs)
        end_time = time.time()
        inference_time.append(end_time - start_time)

    mean_inference_time = np.mean(np.array(inference_time)[1:])
    return mean_inference_time


def calculate_flop(test_dataset, latent_dim, scenario):
    # Reset graph before testing
    test_cvae = reset_and_load_models(latent_dim, scenario)

    sample = next(iter(test_dataset))
    enc_sample = tf.constant(np.expand_dims(sample[0], axis=0))

    encoder_flops = flops.get_flops(test_cvae.encoder, [enc_sample])

    dec_sample = tf.constant(np.expand_dims(test_cvae.encoder(sample)[0, ..., :latent_dim], axis=0))
    decoder_flops = flops.get_flops(test_cvae.decoder, [dec_sample])

    total_flops = encoder_flops + decoder_flops

    return total_flops


def calculate_params(latent_dim, scenario):
    # Reset graph before testing
    test_cvae = reset_and_load_models(latent_dim, scenario)

    enc_trainable_params = sum(count_params(layer) for layer in test_cvae.encoder.trainable_weights)
    enc_non_trainable_params = sum(count_params(layer) for layer in test_cvae.encoder.non_trainable_weights)
    dec_trainable_params = sum(count_params(layer) for layer in test_cvae.decoder.trainable_weights)
    dec_non_trainable_params = sum(count_params(layer) for layer in test_cvae.decoder.non_trainable_weights)

    total_params = enc_trainable_params + enc_non_trainable_params + dec_trainable_params + dec_non_trainable_params

    return total_params


def save_and_calculate_metrics(cvae, test_dataset, scenario):
    # Save
    save_models(cvae.encoder, cvae.decoder, scenario)

    # Time
    mean_inference_time = calculate_time(test_dataset, cvae.latent_dim, scenario)

    # FLOPs
    total_flops = calculate_flop(test_dataset, cvae.latent_dim, scenario)

    # Parameters
    total_params = calculate_params(cvae.latent_dim, scenario)

    # Reset before exit.
    tf.keras.backend.clear_session()

    return mean_inference_time, total_flops, total_params
