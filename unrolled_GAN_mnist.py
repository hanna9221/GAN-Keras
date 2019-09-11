import numpy as np
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
import time


# Load and normalize dadta
#(X_train, _), (_, _) = mnist.load_data()
#X_train = X_train / 127.5 - 1
#X_train = np.expand_dims(X_train, 3)

# Constants
SEED = 87
np.random.seed(SEED)
random_dim     = 100
batch_size     = 64
unrolling_step = 3
optimizer      = Adam(0.0002, 0.5)
count          = 0
epochs         = 40

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
    
    # output
    model.add(Conv2D(1, kernel_size = 3, activation = 'tanh', padding = 'same', kernel_initializer=init))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    
    return model

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

    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    
    return model

def create_gan_model(discriminator, latent_dim, generator):
    discriminator.trainable = False
    
    gan_input = Input(shape = (latent_dim,))
    generator_output = generator(gan_input)
    
    gan_output = discriminator(generator_output)
    gan_model = Model(inputs = gan_input, outputs = gan_output)
    gan_model.compile(loss = 'binary_crossentropy', optimizer = optimizer)
    
    return gan_model

def generator_input(latent_dim, n_samples):
    input = np.random.randn(latent_dim * n_samples)
    input = input.reshape((n_samples, latent_dim))
    return input

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
    
def plot_loss(d_f, d_r, g, count, epochs):
    plt.figure(figsize = (12, 8))
    plt.plot(d_f, label = 'Discriminator Fake Loss')
    plt.plot(d_r, label = 'Discriminator Real Loss')
    plt.plot(g, label = 'Generator Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot_' + str(count+1) + '-' + str(count+epochs) + '.png')
    return

def train_model(X_train, count, epochs = 1, batch_size = 64):
    batch_count = X_train.shape[0] // batch_size
    stop_counter = 0
    
    try:
        generator = load_model('generator.h5')
        discriminator = load_model('discriminator.h5')
        gan_model = create_gan_model(discriminator, random_dim, generator)
    except:
        print("Create new model.")
        generator = create_generator_model()
        discriminator = create_discriminator_model()
        gan_model = create_gan_model(discriminator, random_dim, generator)

    # Lists for Loss History
    discriminator_fake_hist, discriminator_real_hist, generator_hist = [], [], []
        
    for e in range(epochs):
        start = time.time()
        print('=== Epoch {} ==='.format(e+count+1))
        
        for _ in tqdm(range(batch_count)):
            for i in range(unrolling_step):
                # Train discriminator on Fake Images
                discriminator.trainable = True
                X_fake = generator.predict(generator_input(random_dim, batch_size))
                y_fake = np.random.sample(batch_size) * 0.2
                d_fake_loss = discriminator.train_on_batch(X_fake, y_fake)
    
                # Train discriminator on Real Images
                X_real = X_train[np.random.randint(0, X_train.shape[0], size = batch_size)]
                y_real = np.random.sample(batch_size) * 0.1 + 0.8
                d_real_loss = discriminator.train_on_batch(X_real, y_real)
                
                if i==0:
                    discriminator_fake_hist.append(d_fake_loss)
                    discriminator_real_hist.append(d_real_loss)
                    dis_wts_temp = [K.get_value(w) for w in discriminator.weights]

            # Train generator
            noise = generator_input(random_dim, batch_size)
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            generator_loss = gan_model.train_on_batch(noise, y_gen)
            for w, v in zip(discriminator.weights, dis_wts_temp):
                K.set_value(w, v)

            # Store Loss in Loss History lists
            generator_hist.append(generator_loss)
            
        end = time.time()
        elapsed = end - start
        # Sometimes the running time for an epoch increases to a unbearable amount
        if elapsed > 180:
            stop_counter += 1
            
        # Summarize Image Quality for epochs during training
        if (e+count+1) % 20 == 0:
            plot_generated_images(e+count+1, generator)
        
        # Stop training
        if stop_counter > 2:
            plot_generated_images(e+count+1, generator)
            break
                        
    # Plot Loss and Accuracy during Training
    discriminator_fake_hist = np.array(discriminator_fake_hist)
    discriminator_real_hist = np.array(discriminator_real_hist)
    generator_hist = np.array(generator_hist)
    plot_loss(discriminator_fake_hist, discriminator_real_hist, 
              generator_hist, count, epochs)

    # Save model
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')
    
    return

def generate_images():
    try:
        generator = load_model('generator.h5')
    except:
        print('generator does not exist.')
        return
    
    n_samples = 100
    noise = generator_input(random_dim, n_samples)
    img = generator.predict(noise)
    fig = plt.figure(figsize=(10,10))
    for i in range(n_samples):
        fig.add_subplot(10, 10, i+1)
        plt.imshow(img[i,:,:,0], cmap='gray', interpolation='nearest')
        plt.axis('off')
    plt.savefig('generated_images')
    return

train_model(X_train, count, epochs = epochs, batch_size = batch_size)
