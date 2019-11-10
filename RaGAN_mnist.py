import numpy as np
import os, zipfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.layers.core import Dropout
from keras.layers import Input, BatchNormalization, Dense, Reshape
from keras.layers import UpSampling2D, Flatten, Conv2D, ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image
import time

# Load and normalize data
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1
X_train = np.expand_dims(X_train, 3)

# Settings
SEED = 87
np.random.seed(SEED)
random_dim   = 100
optimizer    = Adam(0.0002, 0.5)
batch_size   = 64
batch_count  = X_train.shape[0] // batch_size
count        = 0
epochs       = 60
e_time_limit = 120


def create_generator_model():
    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    model = Sequential()
    start_shape = 64 * 7 * 7
    model.add(Dense(start_shape, kernel_initializer = init, input_dim = random_dim))
    model.add(Reshape((7, 7, 64)))
    
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(Conv2D(1, kernel_size = 3, activation = 'tanh', padding = 'same', kernel_initializer=init))

    noise = Input(shape=(random_dim,))
    img = model(noise)
    return Model(noise, img)

def create_discriminator_model():
    input_shape = (28, 28, 1)

    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, strides = 2, kernel_initializer = init, input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size = 3, strides = 2, kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size = 3, kernel_initializer = init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = init))

    img = Input(shape=input_shape)
    validity = model(img)
    return Model(img, validity)

def generator_input(latent_dim, n_samples):
    # Generate points in latent space
    noise = np.random.randn(latent_dim * n_samples)

    # Reshape to input batch for the network
    noise = noise.reshape((n_samples, latent_dim))

    return noise

def plot_loss(d, g, count, epochs):
    plt.figure(figsize = (12, 8))
    plt.plot(g, label = 'Generator Loss')
    plt.plot(d, label = 'Discriminator Real Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot_' + str(count+1) + '-' + str(count+epochs) + '.png')
    return

def plot_generated_images(epoch, generator, examples = 20, dim = (4, 5)):
    generated_images = generator.predict(np.random.normal(0, 1, size = [examples, random_dim]))
    generated_images = ((generated_images + 1) * 127.5).astype('uint8')
        
    plt.figure(figsize = (12, 8))
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i,:,:,0], cmap='gray')
        plt.axis('off')
    plt.suptitle('Epoch %d' % epoch, x = 0.5, y = 1.0)
    plt.tight_layout()
    plt.savefig('digits_at_epoch_%d.png' % epoch)
    return

def create_images(generator, number):
    # Create Images.zip
    z = zipfile.PyZipFile('images.zip', mode = 'w')
    for k in range(number):
        # Generate new dogs
        generated_images = generator.predict(np.random.normal(0, 1, size = [1, random_dim]))
        image = Image.fromarray(((generated_images + 1) * 127.5).astype('uint8').reshape(64, 64, 3))

        # Save to zip file  
        f = str(k)+'.png'
        image.save(f, 'PNG')
        z.write(f)
        os.remove(f)
        
        # Plot Status Counter
        if k % 1000 == 0: 
            print(k)
    z.close()

# Load or create models
try:
    generator = load_model('generator.h5')
    discriminator = load_model('discriminator.h5')
except:
    print("Create new model.")
    # Create Generator and Discriminator Models
    generator = create_generator_model()
    discriminator = create_discriminator_model()

# Prepare loss function
Real_image  = Input(shape=X_train.shape[1:])
Noise_input = Input(shape=(random_dim,))
Fake_image  = generator(Noise_input)
D_real_out  = discriminator(Real_image)
D_fake_out  = discriminator(Fake_image)

epsilon = 1e-8
D_real_avg = K.mean(D_real_out, axis=0)
D_fake_avg = K.mean(D_fake_out, axis=0)
D_tilda_r = K.sigmoid(D_real_out - D_fake_avg)
D_tilda_f = K.sigmoid(D_fake_out - D_real_avg)

def loss_D(y_true, y_pred):
    return -K.mean(K.log(D_tilda_r + epsilon), axis=0) - K.mean(K.log(1 - D_tilda_f + epsilon), axis=0)

def loss_G(y_true, y_pred):
    return -K.mean(K.log(D_tilda_f + epsilon), axis=0) - K.mean(K.log(1 - D_tilda_r + epsilon), axis=0)
        
# Compile model
generator_train = Model([Noise_input, Real_image], [D_fake_out, D_real_out])
discriminator.trainable = False
generator_train.compile(loss=[loss_G, None], optimizer=optimizer)

discriminator_train = Model([Noise_input, Real_image], [D_real_out, D_fake_out])
generator.trainable = False
discriminator.trainable = True
discriminator_train.compile(loss=[loss_D, None], optimizer=optimizer)


# === Train ===
# Lists for Loss History
discriminator_hist, generator_hist = [], []

stop_counter = 0
dummy_y = np.zeros((batch_size, 1))
for e in range(epochs):
    start = time.time()
    print('=== Epoch {} ==='.format(e+count+1))
    for _ in tqdm(range(batch_count)):
        
        # Train discriminator
        discriminator.trainable = True
        generator.trainable = False
        noise = generator_input(random_dim, batch_size)
        img_batch = X_train[np.random.randint(0, X_train.shape[0], size = batch_size)]
        d_loss = discriminator_train.train_on_batch([noise, img_batch], dummy_y)
        discriminator_hist.append(d_loss)

        # Train generator
        discriminator.trainable = False
        generator.trainable = True
        noise = generator_input(random_dim, batch_size)
        img_batch = X_train[np.random.randint(0, X_train.shape[0], size = batch_size)]
        g_loss = generator_train.train_on_batch([noise, img_batch], dummy_y)

        # Store Loss in Loss History lists
        generator_hist.append(g_loss)
        
    end = time.time()
    elapsed = end - start
    # Sometimes the running time for an epoch increases to a unbearable amount
    if elapsed > e_time_limit:
        stop_counter += 1
        
    # Summarize Image Quality for epochs during training
    if (e+count+1) % 20 == 0:
        plot_generated_images(e+count+1, generator)
    
    # Stop training
    if stop_counter > 2:
        plot_generated_images(e+count+1, generator)
        break
                    
# Plot Loss during Training
plot_loss(discriminator_hist, generator_hist, count, epochs)

# Save model
generator.save('generator.h5')
discriminator.save('discriminator.h5')

