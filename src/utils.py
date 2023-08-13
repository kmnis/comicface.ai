import os

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.utils import array_to_img


def get_dirs(save_dir):
    checkpoint_dir = os.path.join(save_dir, "model_ckpts/ckpts")
    log_dir = os.path.join(save_dir, "tf_logs")
    save_path = os.path.join(save_dir, "training_progress")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    return checkpoint_dir, log_dir, save_path


def generate_images(model, test_input, target, save_path, step):
    prediction = model(test_input)
    plt.figure(figsize=(9, 3))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    try:
        img = array_to_img(prediction[0] * 0.5 + 0.5)
        img.save(f'{save_path}/{step}.png')
    except Exception as e:
        print(e)
        pass