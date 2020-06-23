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

    def __init__(self, data_file_name):
        # hyperparameters
        self.hidden_size = 100  # size of hidden layer of neurons
        self.seq_length = 25  # number of steps to unroll the RNN for
        self.learning_rate = 1e-1

        self.data, self.vocab_size, self.char_to_ix, self.ix_to_char = self.read_data_file(data_file_name)

        # model parameters
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((self.hidden_size, 1))  # hidden bias
        self.by = np.zeros((self.vocab_size, 1))  # output bias

    def save_parameters(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        np.savetxt(os.path.join(folder_path, 'Wxh.csv'), self.Wxh)
        np.savetxt(os.path.join(folder_path, 'Whh.csv'), self.Whh)
        np.savetxt(os.path.join(folder_path, 'Why.csv'), self.Why)
        np.savetxt(os.path.join(folder_path, 'bh.csv'), self.bh)
        np.savetxt(os.path.join(folder_path, 'by.csv'), self.by)

    def load_parameters(self, folder_path):
        self.Wxh = np.loadtxt(os.path.join(folder_path, 'Wxh.csv'))
        self.Whh = np.loadtxt(os.path.join(folder_path, 'Whh.csv'))
        self.Why = np.loadtxt(os.path.join(folder_path, 'Why.csv'))
        self.bh = np.loadtxt(os.path.join(folder_path, 'bh.csv')).reshape(-1,1)
        self.by = np.loadtxt(os.path.join(folder_path, 'by.csv')).reshape(-1,1)

    def read_data_file(self, data_file_name):

        # data I/O
        data = open(data_file_name, 'r').read()  # should be simple plain text file

        # Randomise order of words in data, for training
        words = data.split("\n")
        random.shuffle(words)
        data = "\n".join(words)

        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
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

    def train(self, print_every=100, save_every=10_000):
        loss_over_time_x = []
        loss_over_time_y = []

        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        # mWxh, mWhh, mWhy = np.copy(self.Wxh), np.copy(self.Whh), np.copy(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        # mbh, mby = np.copy(self.bh), np.copy(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0

        while True:
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
                print('----\n %s \n----' % (txt,))
                # Progress
                print('iter %d, loss: %f' % (n, smooth_loss))
                # Chart of loss over time
                loss_over_time_x.append(n)
                loss_over_time_y.append(smooth_loss)
                plt.clf()
                plt.plot(loss_over_time_x, loss_over_time_y, label='loss')
                plt.legend()
                plt.savefig('loss.png')
            # Save weights
            if n % save_every == 0:
                self.save_parameters(os.path.join('rnn_weights', 'iteration_{0}'.format(n)))

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += self.seq_length  # move data pointer
            n += 1  # iteration counter


if __name__ == '__main__':
    nn = RecurrentNeuralNet(data_file_name='test-gb-places.txt')
    # nn.load_parameters(os.path.join('rnn_weights', 'iteration_38000'))
    nn.train(print_every=1_000, save_every=100_000)
