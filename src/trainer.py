import os
import time
from datetime import datetime
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import array_to_img

from .networks import pix2pix_generator, pix2pix_discriminator, vae_encoder, vae_decoder
from .data_loader import data_loader, BATCH_SIZE
from .losses import generator_loss, discriminator_loss
from .utils import generate_images, get_dirs


def tf_logs(log_dir):
    summary_writer = tf.summary.create_file_writer(
        log_dir + "/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return summary_writer


class Pix2PixModel(object):
    """Pix2pix class.
  
    Args:
      epochs: Number of epochs.
      enable_function: If true, train step is decorated with tf.function.
      buffer_size: Shuffle buffer size..
      batch_size: Batch size.
    """

    def __init__(self, epochs=5, enable_function=True, save_dir="../saved_models/pix2pix"):
        self.epochs = epochs
        self.enable_function = enable_function
        self.lambda_value = 100

        self.save_dir = save_dir
        self.checkpoint_dir, self.log_dir, self.save_path = get_dirs(self.save_dir)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.generator = pix2pix_generator()
        self.discriminator = pix2pix_discriminator()

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.summary_writer = tf_logs(self.log_dir)

    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_loss = generator_loss(self.loss_object, disc_generated_output, gen_output, target, self.lambda_value)
            disc_loss = discriminator_loss(self.loss_object, disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, data=None):
        if self.enable_function:
            self.train_step = tf.function(self.train_step)

        if data is None:
            train_ds, test_ds = data_loader()
        else:
            train_ds, test_ds = data

        example_input, example_target = next(iter(test_ds.take(1)))
        start = time.time()

        steps = int(9000 / BATCH_SIZE) * self.epochs
        pbar = tqdm(total=steps)
        step = 0
        for epoch in range(self.epochs):
            for input_image, target_image in train_ds:
                if step % 100 == 0:
                    generate_images(self.generator, example_input, example_target, self.save_path, step)

                gen_loss, disc_loss = self.train_step(input_image, target_image)

                with self.summary_writer.as_default():
                    tf.summary.scalar('generator_loss', gen_loss, step=step)
                    tf.summary.scalar('discriminator_loss', disc_loss, step=step)

                # Save (checkpoint) the model every 5k steps
                if (step + 1) % 500 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_dir)
                    self.generator.save(f"{self.save_dir}/pix2pix.keras")

                step += 1
                pbar.update()

            self.generator.save(f"{self.save_dir}/pix2pix.keras")

        self.generator.save(f"{self.save_dir}/pix2pix.keras")
        return self.generator


class VAEMonitor(Callback):
    def __init__(self, sample_content, sample_style, save_path="../saved_models/vae/training_progress/"):
        self.sample_content = sample_content
        self.sample_style = sample_style
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):

        pred = self.model.predict(self.sample_content)

        if epoch % 10 == 0:
            # Plot the Style, Content and the NST image.
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
            ax[0].imshow(array_to_img(self.sample_content[0]))
            ax[0].set_title("Input Image")

            ax[1].imshow(array_to_img(self.sample_style[0]))
            ax[1].set_title("Ground Truth")

            ax[2].imshow(array_to_img(pred[0]))
            ax[2].set_title("Predicted Image")

            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.close()

        try:
            img = array_to_img(pred[0])
            img.save(f'{self.save_path}/{epoch}.png')
        except Exception as e:
            print(e)
            pass


class CVAE():
    def __init__(self, latent_dim=256):
        self.latent_dim = latent_dim
        self.encoder = vae_encoder(self.latent_dim)
        self.decoder = vae_decoder(self.latent_dim)
        self.callback = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
        self.model = self.build_vae()
        
    def build_vae(self):
        encoder_input = Input(shape=(256, 256, 3))
        z_mean, z_log_var, z = self.encoder(encoder_input)
        vae_output = self.decoder(z)
        vae = Model(encoder_input, vae_output, name='vae')
        return vae
    
    def train(self, data=None, epochs=100, save_path="../saved_models/vae/vae.keras"):
        
        if data is None:
            train_ds, test_ds = data_loader()
        else:
            train_ds, test_ds = data
        
        example_input, example_target = next(iter(test_ds.take(1)))
        train_monitor = VAEMonitor(example_input, example_target)
        
        # Build and compile the VAE model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the VAE model with validation
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=[self.callback, train_monitor]
        )
        
        self.model.save(save_path)
        return history, self.model


if __name__ == "__main__":
    # Train Pix2Pix model
    model = Pix2PixModel()
    model.train()
    
    # Train VAE model
    # vae_model = CVAE()
    # model = vae_model.train()
