import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import kagglehub
import os
import cv2
import random

from tensorflow import keras

from dcgan import latent, generator, DCGAN
from config import get_device


get_device(use_apple_gpu=True)

# dataset
path = kagglehub.dataset_download("vishalsubbiah/pokemon-images-and-types")
print("Dataset in:", path)

img_dir = os.path.join(path, "images") # cartella images

print(f"Path directory: {img_dir}")

img_paths = [ # path delle singole immagini
    os.path.join(img_dir, f) 
    for f in os.listdir(img_dir)
]

images = []

for path in img_paths:
    bgr = cv2.imread(path) # lettura immagine cv2 (BGR)
    if bgr is None:
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # conversione RGB
    images.append(rgb)

print(f"Dimensione immagini: {images[0].shape}")

img_stack = np.stack(images, axis=0).astype("float32") # generazione stack per train
img_stack = (img_stack - 127.5) / 127.5 # normalizzazione


def plot_images(number):
    plt.figure(figsize=(10, 5)) # plot esempi

    for i in range(number):
        rnd = random.randrange(len(img_paths))

        plt.subplot(2, 3, i+1)
        plt.imshow(images[rnd])
        plt.axis("off")


def generate_images(number):
    plt.figure(figsize=(10, 5)) # plot generate

    generated = generator(tf.random.normal([number, latent]), training=False)

    for i in range(len(generated)):
        plt.subplot(2, 3, i+1)
        plt.imshow(generated[i])
        plt.axis("off")


# callback (standard Keras) per visualizzazione risultati
class ShowImg(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 20 == 0: # mostra ogni 5 epoche
            preds = generator(tf.random.normal([16, latent]), training=False) # genera 16 immagini 
            plt.figure(figsize=(4, 4))

            # plotting
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow((preds[i] * 127.5 + 127.5).numpy().astype("uint8"), cmap='gray')
                plt.axis('off')
            plt.suptitle(f"Epoca: {epoch+1}"); plt.show()


dcgan = DCGAN()
dcgan.compile(g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
            d_opt = keras.optimizers.Adam(4e-5, beta_1=0.5, beta_2=0.999),  # discriminatore più lento 
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
dcgan.fit(img_stack, epochs=200, callbacks=[ShowImg()])

plot_images(6) # immagini originali
generate_images(6) # immagini generate