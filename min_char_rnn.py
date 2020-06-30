"""
Adapted from: https://gist.github.com/karpathy/d4dee566867f8291f086
              Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
              BSD License
"""
import random
import os.path
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class RecurrentNeuralNet:

    @classmethod
    def from_saved(cls, save_folder, iteration_number):
        """
        Returns: A RecurrentNeuralNet instance, loaded from save_folder, at iteration_number.
        Assumes: iteration_number is a point at which the rnn in save_folder was saved at.
        """
        rnn = RecurrentNeuralNet(data_file_name=os.path.join(save_folder, 'data.txt'), save_folder='temp')
        rnn.save_folder = save_folder
        rnn.load_parameters(iteration_number)
        return rnn

    def __init__(self, data_file_name, save_folder='rnn'):
        """
        Creates a new, untrained, RecurrentNeuralNet instance.
        Args:
            data_file_name: Name of file to use as data for future training.
            save_folder: Name of folder into which to save data about this RecurrentNeuralNet.
        """
        # hyperparameters
        self.hidden_size = 100  # size of hidden layer of neurons
        self.seq_length = 25  # number of steps to unroll the RNN for
        self.learning_rate = 1e-1

        self.data, self.vocab_size, self.char_to_ix, self.ix_to_char = self.read_data_file(data_file_name)
        self.all_real_words = set(self.data.split('\n'))
        self.all_real_words.discard('')

        # model parameters
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((self.hidden_size, 1))  # hidden bias
        self.by = np.zeros((self.vocab_size, 1))  # output bias

        # save data to file
        self.save_folder = save_folder
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(save_folder, 'data.txt'), 'w') as file:
            file.write(self.data)

    def save_parameters(self, iteration_number):
        folder_path = os.path.join(self.save_folder, 'weights', 'iteration_{0}'.format(iteration_number))
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(folder_path, 'Wxh.csv'), self.Wxh)
        np.savetxt(os.path.join(folder_path, 'Whh.csv'), self.Whh)
        np.savetxt(os.path.join(folder_path, 'Why.csv'), self.Why)
        np.savetxt(os.path.join(folder_path, 'bh.csv'), self.bh)
        np.savetxt(os.path.join(folder_path, 'by.csv'), self.by)

    def load_parameters(self, iteration_number):
        with open(os.path.join(self.save_folder, 'data.txt'), 'r') as file:
            self.data = file.read()
        folder_path = os.path.join(self.save_folder, 'weights', 'iteration_{0}'.format(iteration_number))
        self.Wxh = np.loadtxt(os.path.join(folder_path, 'Wxh.csv'))
        self.Whh = np.loadtxt(os.path.join(folder_path, 'Whh.csv'))
        self.Why = np.loadtxt(os.path.join(folder_path, 'Why.csv'))
        self.bh = np.loadtxt(os.path.join(folder_path, 'bh.csv')).reshape(-1, 1)
        self.by = np.loadtxt(os.path.join(folder_path, 'by.csv')).reshape(-1, 1)

    def read_data_file(self, data_file_name):

        # data I/O
        data = open(data_file_name, 'r').read()  # should be simple plain text file

        # Randomise order of words in data, for training
        words = data.split("\n")
        words = list(set(words))
        random.shuffle(words)
        data = "\n".join(words)

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}

        return data, vocab_size, char_to_ix, ix_to_char

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def prime_h_from_data(self, num_chars_min=500):
        """
        Generates a value for h, by priming the rnn with real data.
        Args:
            num_chars_min: Minimum number of data characters to prime with. Continues from this number to the next newline character.

        Returns: h value
        """
        h = np.zeros((self.hidden_size, 1))

        # prime with real data, up to the next newline character
        i = 0
        current_char = self.data[0]
        while current_char != '\n' or i < num_chars_min:
            ix = self.char_to_ix[current_char]
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            i += 1
            current_char = self.data[i]

        return h

    def get_one_word(self, h=None):
        # if no h provided, then prime with real data
        if h is None:
            h = self.prime_h_from_data()

        # generate a new word, until we reach a newline character
        word = ''
        current_char = '\n'
        ix = self.char_to_ix[current_char]
        x = np.zeros((self.vocab_size, 1))
        x[ix] = 1
        while current_char != '\n' or word == '':
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            current_char = self.ix_to_char[ix]
            word += current_char

        # remove newline character from word
        word = word.strip()
        return word, h

    def train(self, num_iterations=100_000_000, print_every=100, save_every=10_000):
        loss_over_time_x = []
        loss_over_time_y = []

        p = 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        # mWxh, mWhh, mWhy = np.copy(self.Wxh), np.copy(self.Whh), np.copy(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        # mbh, mby = np.copy(self.bh), np.copy(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0

        for n in range(num_iterations):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.seq_length + 1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))  # reset RNN memory
                p = 0  # go from start of data
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p + self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + self.seq_length + 1]]

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # print progress
            if n % print_every == 0:
                # Random samples of output
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print(txt.replace("\n", '|'))
                # Progress
                print('iter %d, loss: %f' % (n, smooth_loss))
                # Chart of loss over time
                loss_over_time_x.append(n)
                loss_over_time_y.append(smooth_loss)
                plt.clf()
                plt.plot(loss_over_time_x, loss_over_time_y, label='loss')
                plt.legend()
                plt.savefig(os.path.join(self.save_folder, 'loss.png'))
            # Save weights
            if n % save_every == 0:
                self.save_parameters(n)

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += self.seq_length  # move data pointer


def train_from_scratch(data_file, save_folder, print_every, save_every):
    nn = RecurrentNeuralNet(data_file_name=data_file, save_folder=save_folder)
    nn.train(print_every=print_every, save_every=save_every)


def print_from_saved(save_folder, iteration_number):
    nn = RecurrentNeuralNet.from_saved(save_folder=save_folder, iteration_number=iteration_number)

    h = np.zeros((nn.hidden_size, 1))
    sample_ix = nn.sample(h, nn.char_to_ix["\n"], 200)
    text = ''.join(nn.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))


def test_print_words_from_saved(save_folder, iteration_number, num_words=10):
    nn = RecurrentNeuralNet.from_saved(save_folder=save_folder, iteration_number=iteration_number)

    print('--- From data initialisation')
    for i in range(num_words):
        word, h = nn.get_one_word()
        print(word)

    print('\n--- From own h')
    for i in range(num_words):
        word, h = nn.get_one_word(h)
        print(word)


def generator_fake_words(save_folder, iteration_number, num_words=10):
    nn = RecurrentNeuralNet.from_saved(save_folder=save_folder, iteration_number=iteration_number)

    # get a h value from looking at the data
    h = nn.prime_h_from_data()

    # run nn, reusing h, making sure words are not duplicates of real words
    i = 0
    while i < num_words:
        word, h = nn.get_one_word(h)

        if word not in nn.all_real_words:
            yield word
            i += 1


def generate_fake_words(save_folder, iteration_number, num_words=10):
    return list(generator_fake_words(save_folder, iteration_number, num_words))


def generate_real_words(save_folder, num_words=10):
    nn = RecurrentNeuralNet.from_saved(save_folder=save_folder, iteration_number=0)

    words = list(np.random.choice(list(nn.all_real_words), num_words, replace=False))
    return words


def train_then_print_check(data_file):
    nn = RecurrentNeuralNet(data_file_name=data_file)
    nn.train(num_iterations=6_000, print_every=1_000, save_every=50000000)
    print('===========================================================')
    print()

    nn.debug = True

    print('In memory after training, zero h, first input \\n')
    h = np.zeros((nn.hidden_size, 1))
    sample_ix = nn.sample(h, nn.char_to_ix["\n"], 200)
    text = ''.join(nn.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))

    h = np.zeros((nn.hidden_size, 1))
    sample_ix = nn.sample(h, nn.char_to_ix["\n"], 200)
    text = ''.join(nn.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))
    print()

    print('In memory, after save & load (save with zero h), zero h, first input \\n')
    h = np.zeros((nn.hidden_size, 1))
    nn.save_parameters('test_weights')
    nn.load_parameters('test_weights')
    sample_ix = nn.sample(h, nn.char_to_ix["\n"], 200)
    text = ''.join(nn.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))

    h = np.zeros((nn.hidden_size, 1))
    sample_ix = nn.sample(h, nn.char_to_ix["\n"], 200)
    text = ''.join(nn.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))
    print()

    print('New network, after load, zero h, first input \\n')
    nn_2 = RecurrentNeuralNet(data_file_name=data_file)
    nn_2.debug = True
    nn_2.load_parameters('test_weights')
    h = np.zeros((nn_2.hidden_size, 1))
    sample_ix = nn_2.sample(h, nn_2.char_to_ix["\n"], 200)
    text = ''.join(nn_2.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))

    h = np.zeros((nn.hidden_size, 1))
    sample_ix = nn_2.sample(h, nn_2.char_to_ix["\n"], 200)
    text = ''.join(nn_2.ix_to_char[ix] for ix in sample_ix)
    print(text.replace("\n", '|'))
    print()


def generate_fake_words_to_file(save_folder, iteration_number, num_words, file_name):
    with open(file_name, 'w') as file:
        file.write('{0} = [\n'.format(save_folder))
        for word in generate_fake_words(save_folder, iteration_number, num_words):
            file.write("'{0}', \n".format(word.replace("'", "\\'")))
        file.write(']\n')


def generate_real_words_to_file(save_folder, file_name):
    nn = RecurrentNeuralNet.from_saved(save_folder=save_folder, iteration_number=0)
    with open(file_name, 'w') as file:
        variable_name = save_folder
        if variable_name.startswith('rnn_'):
            variable_name = 'real_' + variable_name[len('rnn_'):]
        file.write('{0} = [\n'.format(variable_name))
        for word in sorted(nn.all_real_words):
            file.write("'{0}', \n".format(word.replace("'", "\\'")))
        file.write(']\n')


if __name__ == '__main__':
    # train_from_scratch(data_file='scottish-places.txt', save_folder='16_scottish_places', print_every=10_000, save_every=100_000)
    # train_from_scratch(data_file='welsh-places.txt', save_folder='17_welsh_places', print_every=10_000, save_every=100_000)
    # train_from_scratch(data_file='irish-places.txt', save_folder='rnn_irish_places', print_every=10_000, save_every=100_000)
    # train_from_scratch(data_file='french-places.txt', save_folder='rnn_french_places', print_every=10_000, save_every=100_000)
    # print_from_saved(saved_folder=os.path.join('13_overnight_run_rnn_weights', 'iteration_0'))
    # print_from_saved(saved_folder=os.path.join('13_overnight_run_rnn_weights', 'iteration_100000'))
    # print_from_saved(saved_folder=os.path.join('13_overnight_run_rnn_weights', 'iteration_11500000'))
    # print_from_saved(saved_folder=os.path.join('rnn_weights', 'iteration_7000'))
    # train_then_print_check()
    # print_from_saved('test_weights')
    # iteration_number = 1_325_000  # 400_000
    # print_from_saved(save_folder='14_fixed_loading', iteration_number=iteration_number)
    # test_print_words_from_saved(save_folder='14_fixed_loading', iteration_number=iteration_number)
    # test_print_words_from_saved(save_folder='15_very_long_training', iteration_number=12_600_000)
    # num_words = 5
    # print(generate_fake_words(save_folder='15_very_long_training', iteration_number=12_600_000, num_words=num_words))
    # print(generate_real_words(save_folder='15_very_long_training', num_words=num_words))
    # generate_fake_words_to_file(save_folder='rnn_english_places', iteration_number=12_600_000, num_words=20_000, file_name='fake_english_places.js')
    # generate_real_words_to_file('rnn_english_places', 'real_english_places.js')
    # generate_fake_words_to_file(save_folder='rnn_scottish_places', iteration_number=15_900_000, num_words=20_000, file_name='fake_scottish_places.js')
    # generate_real_words_to_file('rnn_scottish_places', 'real_scottish_places.js')
    # generate_fake_words_to_file(save_folder='rnn_welsh_places', iteration_number=15_900_000, num_words=20_000, file_name='fake_welsh_places.js')
    # generate_real_words_to_file('rnn_welsh_places', 'real_welsh_places.js')
    generate_fake_words_to_file(save_folder='rnn_irish_places', iteration_number=14_800_000, num_words=20_000, file_name='fake_irish_places.js')
    generate_real_words_to_file('rnn_irish_places', 'real_irish_places.js')
    generate_fake_words_to_file(save_folder='rnn_french_places', iteration_number=14_200_000, num_words=20_000, file_name='fake_french_places.js')
    generate_real_words_to_file('rnn_french_places', 'real_french_places.js')
