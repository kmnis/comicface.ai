from glob import glob
from PIL import Image

model = "vae" # pix2pix
image_files = glob(f"saved_models/{model}/training_progress/*.png")

def numeric_sort_key(filename):
    return int(filename.split("/")[-1][:-4])
    
image_files.sort(key=numeric_sort_key)

print("Total images found:", len(image_files))

images = [Image.open(f) for f in image_files]

if model == "pix2pix":
    images = images[0:20] + images[25:30]
    durations = [150] * (len(images) - 1) + [1000]
else:
    durations = [50] * (len(images) - 1) + [1000]

images[0].save(f"saved_models/{model}/training_progress/{model}_training.gif", save_all=True, append_images=images[1:], duration=durations, loop=0)
