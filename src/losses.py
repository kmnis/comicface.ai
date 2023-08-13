import tensorflow as tf


def generator_loss(loss_object, disc_generated_output, gen_output, target, lambda_value):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (lambda_value * l1_loss)

    return total_gen_loss


def discriminator_loss(loss_object, disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    return real_loss + generated_loss