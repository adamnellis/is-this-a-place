import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


latent_space_size = 5


def load_data():
    word_list, alphabet = read_in_words('test-gb-places.txt')
    max_word_length = max(len(word) for word in word_list)

    words_as_numbers = [word_to_numbers(word, max_word_length, alphabet) for word in word_list]

    training_words = np.asarray(words_as_numbers[::2])
    test_words = np.asarray(words_as_numbers[1::2])
    training_classes = np.zeros(len(training_words))
    test_classes = np.zeros(len(test_words))

    return training_words, training_classes, test_words, test_classes, max_word_length, alphabet


def word_to_numbers(word, word_length, alphabet):
    temp = [alphabet.index(letter) / word_length for letter in word]
    return temp + [0] * (word_length - len(temp))


def numbers_to_word(numbers, word_length, alphabet):
    word = [''] * len(numbers)
    for i, number in enumerate(numbers):
        try:
            word[i] = alphabet[int(number * word_length)]
        except IndexError:
            pass
    return ''.join(word).strip()


def read_in_words(file_path):
    # Read words into a set
    with open(file_path, 'r') as file:
        word_list = set()
        for line in file.readlines():
            # Clean up each word
            word = line.strip()
            if word.startswith('"') and word.endswith('"'):
                word = word[1:-1]
            word_list.add(word)
    # Extract unique letters used in words
    alphabet = set(letter for word in word_list for letter in word)
    # Turn sets into lists
    word_list = list(sorted(word_list))
    alphabet = list(sorted(alphabet))
    return word_list, alphabet


def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)


def generate_input_noise(batch_size):
    # return np.random.normal(0, 1, [batch_size, latent_space_size])
    return np.random.uniform(0, 1, [batch_size, latent_space_size])
    # return np.zeros([batch_size, latent_space_size])  # makes arbitrary looking words
    # return np.ones([batch_size, latent_space_size])  # evolves towards eeeee
    # return np.ones([batch_size, latent_space_size]) * 60  # starts with a lot of eee
    # return np.ones([batch_size, latent_space_size]) * 0.3  # evolves more slowly towards eeee
    # return np.random.uniform(-0.1, 0.1, [batch_size, latent_space_size])


def create_generator():
    generator = Sequential()

    # Images example
    # generator.add(Dense(units=256, input_dim=100))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=512))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=1024))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=784, activation='tanh'))

    # Images example translated into words
    # generator.add(Dense(units=16, input_dim=10))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=32))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=64))
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Dense(units=37, activation='tanh'))

    # Circles example translated into words
    generator.add(Dense(16, input_dim=latent_space_size, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(37, activation='relu', kernel_initializer='he_uniform'))
    # generator.add(Dense(37, activation='sigmoid'))

    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator


def create_discriminator():
    discriminator = Sequential()

    # Images example
    # discriminator.add(Dense(units=1024, input_dim=784))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    #
    # discriminator.add(Dense(units=512))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    #
    # discriminator.add(Dense(units=256))
    # discriminator.add(LeakyReLU(0.2))

    # Images example translated into words
    # discriminator.add(Dense(units=64, input_dim=37))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    #
    # discriminator.add(Dense(units=32))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))
    #
    # discriminator.add(Dense(units=16))
    # discriminator.add(LeakyReLU(0.2))

    # Circles example translated into words
    discriminator.add(Dense(64, input_dim=37, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_space_size,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def train_network(epochs=1, batch_size=128):
    # Loading the data
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    batch_count = x_train.shape[0] / batch_size

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_size)):
            # generate random noise as an input to initialize the generator
            noise = generate_input_noise(batch_size)

            # Generate fake words from noise input
            generated_images = generator.predict(noise)

            # Get a random set of real images
            image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

            # Construct different batches of real and fake data
            x = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # Pre train discriminator on fake and real data before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(x, y_dis)

            # Tricking the noised input of the Generator as real data
            noise = generate_input_noise(batch_size)
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            # We can enforce that by setting the trainable flag
            discriminator.trainable = False

            # training the GAN by alternating the training of the Discriminator
            # and training the chained GAN model with Discriminator's weights frozen.
            gan.train_on_batch(noise, y_gen)

        # if e == 1 or e % 20 == 0:
        if e == 1 or e % 4 == 0:
            generate_random_words(generator, 6)


def print_summaries():
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    print('Training data shape (num rows, num cols): ', x_train.shape)
    g = create_generator()
    print('Generator summary:')
    g.summary()
    d = create_discriminator()
    print('Discriminator summary:')
    d.summary()
    gan = create_gan(d, g)
    print('GAN summary:')
    gan.summary()


def investigate_generator(batch_size):
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    generator = create_generator()
    print('Generator summary:')
    generator.summary()
    noise = generate_input_noise(batch_size)
    print('Noise: ', noise)
    generated_words = generator.predict(noise)
    generated_words = [numbers_to_word(numbers, max_word_length, alphabet) for numbers in generated_words]
    print('Random words:\n  ' + '\n  '.join(generated_words))


def generate_random_words(generator, batch_size):
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    noise = generate_input_noise(batch_size)
    generated_words = generator.predict(noise)
    generated_words = [numbers_to_word(numbers, max_word_length, alphabet) for numbers in generated_words]
    print('Generated words:\n  ' + '\n  '.join(generated_words))


if __name__ == '__main__':
    print_summaries()
    generate_random_words(generator=create_generator(), batch_size=6)
    train_network(4000, 128)
