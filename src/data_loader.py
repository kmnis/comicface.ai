import os
import tensorflow as tf

# Define Training variable
BUFFER_SIZE = 400
BATCH_SIZE = 32
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.AUTOTUNE


def load_images(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    image = tf.cast(image, tf.float32)
    return image


def resize(content_image, style_image, height, width):
    content_image = tf.image.resize(content_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if style_image is not None:
        style_image = tf.image.resize(style_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return content_image, style_image


def random_crop(content_image, style_image):
    stacked_image = tf.stack([content_image, style_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(content_image, style_image):
    content_image = (content_image / 127.5) - 1
    
    if style_image is not None:
        style_image = (style_image / 127.5) - 1

    return content_image, style_image


@tf.function()
def random_jitter(content_image, style_image):
    # resizing to 286 x 286 x 3
    content_image, style_image = resize(content_image, style_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    content_image, style_image = random_crop(content_image, style_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        content_image = tf.image.flip_left_right(content_image)
        style_image = tf.image.flip_left_right(style_image)

    return content_image, style_image


def preprocess_train_image(content_path, style_path):
    content_image = load_images(content_path)
    style_image = load_images(style_path)

    content_image, style_image = random_jitter(content_image, style_image)
    content_image, style_image = normalize(content_image, style_image)

    return content_image, style_image


def preprocess_test_image(content_path, style_path=None):
    content_image = load_images(content_path)
    
    if style_path is None:
        style_image = None
    else:
        style_image = load_images(style_path)

    content_image, style_image = resize(content_image, style_image,
                                        IMG_HEIGHT, IMG_WIDTH)
    content_image, style_image = normalize(content_image, style_image)

    if style_image is None:
        return content_image
    else:
        return content_image, style_image


def create_image_loader(path):
    images = os.listdir(path)
    images = [os.path.join(path, p) for p in images]
    images.sort()

    # split the images in train and test
    total_images = len(images)
    train = images[: int(0.9 * total_images)]
    test = images[int(0.9 * total_images):]

    # Build the tf.data datasets.
    train_ds = tf.data.Dataset.from_tensor_slices(train)
    test_ds = tf.data.Dataset.from_tensor_slices(test)

    return train_ds, test_ds


def data_loader(content_path="../data/face", style_path="../data/comics"):
    train_content_ds, test_content_ds = create_image_loader(content_path)
    train_style_ds, test_style_ds = create_image_loader(style_path)

    # Zipping the style and content datasets.
    train_ds = (
        tf.data.Dataset.zip((train_content_ds, train_style_ds))
        .map(preprocess_train_image)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.zip((test_content_ds, test_style_ds))
        .map(preprocess_test_image)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return train_ds, test_ds
