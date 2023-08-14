from glob import glob
from PIL import Image

image_files = glob("saved_models/pix2pix/training_progress/*.png")

def numeric_sort_key(filename):
    return int(filename.split("/")[-1][:-4])
    
image_files.sort(key=numeric_sort_key)

print(len(image_files))

images = [Image.open(f) for f in image_files]
images = images[0:20] + images[25:30]

durations = [150] * (len(images) - 1) + [1000]
images[0].save("saved_models/pix2pix/training_progress/demo.gif", save_all=True, append_images=images[1:], duration=durations, loop=0)
