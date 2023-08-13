import time
from datetime import datetime
from tqdm.notebook import tqdm

import tensorflow as tf

from .networks import Pix2PixGenerator, Pix2PixDiscriminator
from .data_loader import data_loader
from .losses import generator_loss, discriminator_loss
from .utils import generate_images, get_dirs


def get_checkpoint(generator_optimizer, discriminator_optimizer, generator, discriminator):
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )
    return checkpoint


def tf_logs(log_dir):
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return summary_writer


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


def train(model=None, data=None, steps=30000, save_dir="../saved_models/pix2pix"):
    checkpoint_dir, log_dir, save_path = get_dirs(save_path)
    
    if data is None:
        train_ds, test_ds = data_loader()
    else:
        train_ds, test_ds = data
    
    if model is None:
        generator = Pix2PixGenerator()
        discriminator = Pix2PixDiscriminator()
    else:
        generator, discriminator = model
    
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    checkpoint_dir = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = get_checkpoint(
        generator_optimizer,
        discriminator_optimizer,
        generator,
        discriminator
    )
    
    summary_writer = tf_logs(log_dir)
    
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    
    pbar = tqdm(total=steps)
    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            if step != 0:
                print(f'Time taken for steps {step-1000} - {step}: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target, save_path, step//1000)
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_dir)
        
        pbar.update()
    
    return generator


if __name__ == "__main__":
    train()