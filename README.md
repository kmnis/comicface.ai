# Turn Your Photos Into A Comic Book

## Table of contents
1. [Directory Structure](#dir)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Results](#results)

## Directory Structure <a name="dir"></a>
```
|__ data: The training dataset will go here. A few sample images are added for reference
|__ src: Model architecture, data loading, and other utility scripts are saved here
|__ notebooks: The notebooks show an end-to-end pipeline to train and infer the models. The notebook names are self-explanatory
|__ saved_models: Trained models
```

## Dataset <a name="dataset"></a>
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2) and contains 10,000 pairs of face and their comic version. Each image is of 1024x1024 dimensions.

![Sample Image](https://github.com/kmnis/comicface-ai/assets/20987291/a1e9c4b7-b89b-482f-9dcd-ac424116ce5e)

## Models <a name="models"></a>
Two different model architectures are tried for training: Convolutional Variational Autoencoder and Pix2Pix GAN.

### Convolutional VAE

<table>
  <tr>
    <td>
      <img src="https://github.com/kmnis/comicface-ai/assets/20987291/8172ff0d-8208-4d88-a646-d880ff61739a" alt="Model Architecture" width="500"/>
      <p align="center"><em>Model Architecture</em></p>
    </td>
    <td>
      <img src="saved_models/vae/training_progress/vae_training.gif" alt="Image 2"/>
      <p align="center"><em>Convolutional VAE Training</em></p>
    </td>
  </tr>
</table>

### Pix2Pix GAN

<table>
  <tr>
    <td>
      <img src="https://github.com/kmnis/comicface-ai/assets/20987291/b2123082-a1f0-4094-adec-4585a3d8a6bb" alt="Model Architecture" width="500"/>
      <p align="center"><em>Model Architecture</em></p>
    </td>
    <td>
      <img src="saved_models/pix2pix/training_progress/pix2pix_training.gif" alt="Image 2"/>
      <p align="center"><em>Pix2Pix GAN Training</em></p>
    </td>
  </tr>
</table>

## Results <a name="results"></a>

<table>
  <tr>
    <td>
      <img src="https://github.com/kmnis/comicface-ai/assets/20987291/0a163dc0-6fb5-4e13-b80e-f7e855bf9055" alt="Model Architecture" width="100%"/>
      <p align="center"><em>Convolutional VAE Sample Results</em></p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/kmnis/comicface-ai/assets/20987291/b919f988-7e60-4f88-96dc-460b7e2767b0" alt="Model Architecture" width="100%"/>
      <p align="center"><em>Pix2Pix GAN Sample Results</em></p>
    </td>
  </tr>
</table>
