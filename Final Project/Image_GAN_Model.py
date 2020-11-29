'''
FaceGen by Andy Lee, U1 Honours Computer Science Student @ McGill University
Final Project for MAIS202
'''

from os import listdir
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from keras.optimizers import Adam

'''
# Image Path
dir = "archive/img_align_celeba/img_align_celeba/"

# Open CSVs
attr = pd.read_csv("archive/list_attr_celeba.csv")

# Open single image, return as nparray
# Resize to 80x80
def open_img(fn):
    img = Image.open(fn)
    img = img.resize((80,80))
    img_as_array = np.asarray(img)
    return img_as_array

# Open all images, return as nparray
# Filter out Glasses and Hats in the process
def open_imgs(dir):
    imgs = []
    i = 0
    for image in listdir(dir):
        if (attr.iloc[i]["Eyeglasses"] == 1 or attr.iloc[i]["Wearing_Hat"] == 1):
            i += 1
            continue
        img = open_img(dir + image)
        imgs.append(img)
        print(i)
        i += 1
    imgs = np.asarray(imgs)
    return imgs

# Generate dataset as array
dataset = open_imgs(dir)
# Save dataset arrayfile
np.save('dataset_processed', dataset)

# Load Dataset Array
dataset = np.load("dataset_processed.npy")

dataset_25000 = dataset[0:25000]
np.save('dataset_processed_25000.npy', dataset_25000)
'''
# Load Dataset Array
def show_img(img_as_array):
    img = Image.fromarray(img_as_array)
    img.show()

# Create discriminator
def discriminator():
    # Create Discriminator and Adam Optimizer
    discriminator_model = Sequential()
    discriminator_model.name = "Discriminator"
    discriminator_opti = Adam(lr=0.0002, beta_1=0.5)
    # Outputs 80x80x128
    discriminator_model.add(Conv2D(128, (5,5), padding='same', input_shape=(80,80,3)))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    # Outputs 40x40x128
    discriminator_model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    # Outputs 20x20x128
    discriminator_model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    # Outputs 10x10x128
    discriminator_model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    # Outputs 5x5x128
    discriminator_model.add(Conv2D(128, (5, 5), strides=(2,2), padding='same'))
    discriminator_model.add(LeakyReLU(alpha=0.2))
    # Flatten and Dense
    discriminator_model.add(Flatten())
    discriminator_model.add(Dropout(0.4))
    discriminator_model.add(Dense(1, activation='sigmoid'))
    # Print Model Summary
    discriminator_model.summary()
    # Compile
    discriminator_model.compile(loss='binary_crossentropy', optimizer=discriminator_opti, metrics=['accuracy'])
    return discriminator_model

# Create Generator
def generator():
    # Create Generator for 5x5x128
    generator_model = Sequential()
    generator_model.name = "Generator"
    generator_model.add(Dense(5*5*128, input_dim=100))
    generator_model.add(LeakyReLU(alpha=0.25))
    generator_model.add(Reshape((5,5,128)))
    # 10x10x128
    generator_model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    generator_model.add(LeakyReLU(alpha=0.25))
    # 20x20x128
    generator_model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    generator_model.add(LeakyReLU(alpha=0.25))
    # 40x40x128
    generator_model.add(Conv2DTranspose(128, (4, 4), strides=(2,2), padding="same"))
    generator_model.add(LeakyReLU(alpha=0.25))
    # 80x80x128
    generator_model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    generator_model.add(LeakyReLU(alpha=0.25))
    # 80x80x3
    generator_model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    generator_model.summary()
    return generator_model

# Create GAN
def gan(generator, discriminator):
    # Create Adam Optimizer
    gan_opti = Adam(lr=0.0002, beta_1=0.5)
    # Discriminator will not be trained here
    discriminator.trainable = False
    # Create GAN model
    gan_model = Sequential()
    gan_model.name = "FaceGenGAN"
    # Add Generator and Discriminator
    gan_model.add(generator)
    gan_model.add(discriminator)
    # Compile
    gan_model.compile(loss='binary_crossentropy', optimizer=gan_opti)
    gan_model.summary()
    return gan_model

# Create a set of real images to be fed into the discriminator
def create_real(data, num_samples):
    # Create array to hold a number of selected real images
    real_images = []
    # Create an array of 1s to hold the 'real' labels
    real_labels = np.ones((num_samples, 1))

    # Randomly select an image and append to the array
    for i in range(num_samples):
        random_index = np.random.randint(0, data.shape[0])
        real_images.append(data[random_index])

    # Convert list into nparray
    real_images = np.array(real_images)

    # Return images and labels
    return real_images, real_labels

# Create a set of fake images to be fed into the discriminator
def create_fake(generator, dim_latent, num_samples):
    # Generate latent points
    latent = create_latent(dim_latent, num_samples)
    # Feed into generator for prediction
    fake_images = generator.predict(latent)
    # Create 0 ('fake') labels
    fake_labels = np.zeros((num_samples, 1))
    # Return images and labels
    return fake_images, fake_labels

# Create latent points to be fed into the generator
def create_latent(dim_latent, num_samples):
    latent = []
    # Draw from random distribution
    for i in range(num_samples):
        latent.append(np.random.randn(dim_latent))
    # Convert latent list to nparray
    latent = np.array(latent)
    return latent

# Train
def train(dim_latent, data, gen, dis, gan, num_epochs):
    batch_size = 128
    images_per_epoch = int(data.shape[0] / batch_size)

    # For each epoch, the batch will consist of half real and half fake images
    half_batch_size = int(batch_size/2)

    # Run through each epoch
    for i in range(num_epochs):
        # Run through each batch
        for j in range(images_per_epoch):
            # Load real images and train discriminator
            real_images, real_labels = create_real(data, half_batch_size)
            dis_loss_from_real = dis.train_on_batch(real_images, real_labels)

            # Generate fake images and train discriminator
            fake_images, fake_labels = create_fake(gen, dim_latent, half_batch_size)
            dis_loss_from_fake = dis.train_on_batch(fake_images, fake_labels)

            # Generate latent points and train generator
            gan_latents = create_latent(dim_latent, batch_size)
            gan_labels = np.ones((batch_size,1))
            gen_loss = gan.train_on_batch(gan_latents,gan_labels)

            # Every 10 batches, save stats
            if (j+1) % 10 == 0:
                d_loss_real.append(dis_loss_from_real[0])
                d_loss_fake.append(dis_loss_from_fake[0])
                g_loss.append(gen_loss)
                r_acc.append(dis_loss_from_real[1])
                f_acc.append(dis_loss_from_fake[1])

            # Every 5 batches print stats
            if (j+1) % 5 == 0:
                print('[Epoch]: %d, [Batch]: %d/%d, [D-Loss Real]: %.3f, [D-Loss Fake]: %.3f, [G-Loss]: %.3f' % (i + 1, j + 1, images_per_epoch, dis_loss_from_real[0], dis_loss_from_fake[0], gen_loss))

        # Every 10 epochs, generate and save 5x5 plot of fake images
        if (i+1) % 10 == 0:
            real_images_display, real_labels_display = create_real(data, 100)
            accuracy_real = dis.evaluate(real_images_display, real_labels_display, verbose=0)
            fake_images_display, fake_labels_display = create_fake(gen, dim_latent, 100)
            accuracy_fake = dis.evaluate(fake_images_display, fake_labels_display, verbose=0)

            print('\nDiscriminator Accuracy')
            print('[Real]: %.0f%%, [Fake]: %.0f%%\n' % (accuracy_real[1]*100, accuracy_fake[1]*100))

            generate_fakes(fake_images_display, i)
            weight_file = 'generator_weights_%03d.h5' % (i + 1)
            gen.save(weight_file)

# Create a 5x5 plot of generated images and save the plot
def generate_fakes(data, epoch):
    data = (data + 1) / 2.0
    pyplot.suptitle("Generated Fakes - Epoch %d" % (epoch + 1))
    for i in range(25):
        pyplot.subplot(5, 5, i+1)
        pyplot.axis('off')
        pyplot.imshow(data[i])

    fn = 'generated_fakes_%03d.png' % (epoch+1)
    pyplot.savefig(fn)
    pyplot.close()

# Load Dataset
dataset_25000 = np.load("dataset_processed_25000.npy")

# Normalize Image
dataset_25000 = dataset_25000.astype('float32')
dataset_25000 = (dataset_25000 - 127.5) / 127.5

# Create Training Set
training_set = dataset_25000[0:10000]

# D-Loss Real
d_loss_real = []

# D-Loss Fake
d_loss_fake = []

# G_Loss
g_loss = []

# Real_Acc
r_acc = []

# Fake_Acc
f_acc = []

# Hyperparameters
dim_latent = 100
num_epochs = 200

# Create models
gen = generator()
dis = discriminator()
gan = gan(gen, dis)

# Train
train(dim_latent, training_set, gen, dis, gan, num_epochs)

# Create plot of discriminator losses and save
pyplot.plot(d_loss_real, label="D-Loss Real")
pyplot.plot(d_loss_fake, label="D-Loss Fake")
pyplot.legend()
pyplot.savefig('d_loss_results.png')
pyplot.close()

# Create plot of generator losses and save
pyplot.plot(g_loss, label="G-Loss")
pyplot.legend()
pyplot.savefig('g_loss_results.png')
pyplot.close()

# Create plot of discriminator accuracies and save
pyplot.plot(r_acc, label="Real Acc")
pyplot.plot(f_acc, label="Fake Acc")
pyplot.legend()
pyplot.savefig('acc_results.png')
pyplot.close()

print("Done!")

'''
Sources:
[GAN for Dummies](https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391)
[Stable GAN Training](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)
[Generating Human Faces](https://www.kdnuggets.com/2020/03/generate-realistic-human-face-using-gan.html)
[GANs using Keras](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
[Generative Adversarial Network](https://en.wikipedia.org/wiki/Generative_adversarial_network)
[Generating Artificial Faces](https://towardsdatascience.com/face-generator-generating-artificial-faces-with-machine-learning-9e8c3d6c1ead)
[Generating Artificial Faces 2](https://medium.com/coloredfeather/generating-human-faces-using-adversarial-network-960863bc1deb)
'''