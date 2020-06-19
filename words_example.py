import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


latent_space_size = 6  #20


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


def generate_latent_points(batch_size):
    # return np.random.normal(0, 1, [batch_size, latent_space_size])
    return np.random.uniform(0, 1, [batch_size, latent_space_size])
    # return np.zeros([batch_size, latent_space_size])  # makes arbitrary looking words
    # return np.ones([batch_size, latent_space_size])  # evolves towards eeeee
    # return np.ones([batch_size, latent_space_size]) * 60  # starts with a lot of eee
    # return np.ones([batch_size, latent_space_size]) * 0.3  # evolves more slowly towards eeee
    # return np.random.uniform(-0.1, 0.1, [batch_size, latent_space_size])


def create_generator():
    generator = Sequential()

    generator.add(Dense(16, input_dim=latent_space_size, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    generator.add(Dense(37, activation='relu', kernel_initializer='he_uniform'))
    # last layer can't be sigmoid, as that makes the "eeeeee eee e" bug
    # last layer wants to be relu rather than linear, so it can make spaces easily

    return generator


def create_discriminator():
    discriminator = Sequential()

    discriminator.add(Dense(250, input_dim=37, activation='relu', kernel_initializer='he_uniform'))
    # discriminator.add(Dense(250, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(125, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(60, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(30, activation='relu', kernel_initializer='he_uniform'))
    discriminator.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator


def create_gan(discriminator, generator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False

    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)

    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan


def train_network(epochs=1, batch_size=128, print_every=400, figure_every=4000):
    # Loading the data
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()

    # Creating GAN
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    def generate_real_samples(n):
        samples = x_train[np.random.randint(low=0, high=x_train.shape[0], size=n)]
        classes = np.ones((n, 1))
        return samples, classes

    def generate_fake_samples(generator, n):
        # generate points in latent space
        x_input = generate_latent_points(n)
        # predict outputs
        samples = generator.predict(x_input)
        # create class labels
        classes = np.zeros((n, 1))
        return samples, classes

    def summarise_performance(epoch, generator, discriminator, n=100):
        # prepare real samples
        x_real, y_real = generate_real_samples(n)
        # evaluate discriminator on real examples
        acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(generator, n)
        # evaluate discriminator on fake examples
        acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('Epoch: {0}, accuracy(real) = {1:.2f}, accuracy(fake) = {2:.2f}'.format(epoch, acc_real, acc_fake))

        # print some of the words from the generator
        generated_words = [numbers_to_word(numbers, max_word_length, alphabet) for numbers in x_fake[:6]]
        print('Generated words:\n  ' + '\n  '.join(generated_words))

        return acc_real, acc_fake

    # determine half the size of one batch, for updating the discriminator
    half_batch = int(batch_size / 2)

    plot_epochs = []
    plot_accuracy_reals = []
    plot_accuracy_fakes = []

    for epoch in range(epochs):
        # for _ in tqdm(range(batch_size)):
        if True:

            # prepare real samples
            x_real, y_real = generate_real_samples(half_batch)
            # prepare fake examples
            x_fake, y_fake = generate_fake_samples(generator, half_batch)
            # update discriminator
            discriminator.train_on_batch(x_real, y_real)
            discriminator.train_on_batch(x_fake, y_fake)

            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(batch_size)
            # create inverted labels for the fake samples
            y_gan = np.ones((batch_size, 1))
            # update the generator via the discriminator's error
            gan.train_on_batch(x_gan, y_gan)

        if epoch % print_every == 0:
            acc_real, acc_fake = summarise_performance(epoch, generator, discriminator)

            # make plot of accuracy over time
            plot_epochs.append(epoch)
            plot_accuracy_reals.append(acc_real)
            plot_accuracy_fakes.append(acc_fake)
            if epoch % figure_every == 0:
                plt.clf()
                plt.plot(plot_epochs, plot_accuracy_reals, label='accuracy(real)')
                plt.plot(plot_epochs, plot_accuracy_fakes, label='accuracy(fake)')
                plt.legend()
                plt.savefig('accuracy.png')


def print_summaries():
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    print('Max word length:', max_word_length)
    print('Alphabet size:', len(alphabet))
    print('Alphabet:', str(alphabet))
    print('Training data shape (num rows, num cols):', x_train.shape)
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
    noise = generate_latent_points(batch_size)
    print('Noise: ', noise)
    generated_words = generator.predict(noise)
    generated_words = [numbers_to_word(numbers, max_word_length, alphabet) for numbers in generated_words]
    print('Random words:\n  ' + '\n  '.join(generated_words))


def generate_random_words(generator, batch_size):
    (x_train, y_train, x_test, y_test, max_word_length, alphabet) = load_data()
    noise = generate_latent_points(batch_size)
    generated_words = generator.predict(noise)
    generated_words = [numbers_to_word(numbers, max_word_length, alphabet) for numbers in generated_words]
    print('Generated words:\n  ' + '\n  '.join(generated_words))


if __name__ == '__main__':
    print_summaries()
    # generate_random_words(generator=create_generator(), batch_size=6)
    train_network(epochs=400_000, batch_size=128, print_every=400, figure_every=4000)
