import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, backend as K, losses
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from PIL import Image

# Seed for reproducibility
np.random.seed(25)
import tensorflow as tf
tf.random.set_seed(25)

# Load and preprocess custom images
def load_images_from_folder(folder, size=(28, 28)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)  # Resize image
            img = img / 255.0  # Normalize images to [0, 1]
            images.append(img)
    return np.array(images)

# Load your images
image_folder = 'path_to_your_image_folder'  # Update this path
images = load_images_from_folder(image_folder)
images = images.reshape(-1, 28, 28, 1)  # Assuming using grayscale images

# Split data into training and validation sets
X_train, X_test = train_test_split(images, test_size=0.2, random_state=25)

# Define the VAE model architecture
latent_dim = 2  # Latent dimensionality

# Encoder network
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Use reparameterization trick to push the sampling out as input
z = layers.Lambda(lambda x: x[0] + K.exp(x[1] / 2) * K.random_normal(shape=(K.shape(x[0])[0], latent_dim)), output_shape=(latent_dim,))([z_mean, z_log_var])

# Instantiate the encoder
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Decoder network
latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu')(x)
x = layers.Conv2DTranspose(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu')(x)
outputs = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)

# Instantiate the decoder
decoder = models.Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# VAE model
vae_outputs = decoder(encoder(inputs)[2])
vae = models.Model(inputs, vae_outputs, name='vae_mlp')

# Loss Function
reconstruction_loss = losses.binary_crossentropy(K.flatten(inputs), K.flatten(vae_outputs))
reconstruction_loss *= 28 * 28
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# Train the VAE
vae.fit(X_train, epochs=30, batch_size=32, validation_data=(X_test, None))

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        plt.axis('off')

        # Save image
        img = Image.fromarray((predictions[i, :, :, 0] * 255).astype(np.uint8))
        img.save(f'images/image_at_epoch_{str(epoch)}_{i}.png')

    plt.savefig(f'images/image_at_epoch_{str(epoch)}.png')
    plt.show()

# Generate random sample to test generator output
random_latent_vectors = tf.random.normal(shape=(16, latent_dim))
generate_and_save_images(decoder, 0, random_latent_vectors)
