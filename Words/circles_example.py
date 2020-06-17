
#
# Example of using a GAN to do a very simple classification problem
# From: https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/
#


def plot_circles():
    from sklearn.datasets import make_circles
    from numpy import where
    from matplotlib import pyplot
    # generate circles
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # select indices of points with each class label
    for i in range(2):
        samples_ix = where(y == i)
        pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
    pyplot.legend()
    pyplot.show()


def mlp_circles():
    # mlp for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.initializers import RandomUniform
    from matplotlib import pyplot
    # generate 2d classification dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # scale input data to [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    # define model
    model = Sequential()
    init = RandomUniform(minval=0, maxval=1)
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


def deep_mlp_circles():
    # deeper mlp for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.initializers import RandomUniform
    from matplotlib import pyplot
    # generate 2d classification dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    # define model
    init = RandomUniform(minval=0, maxval=1)
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


def deep_mlp_relu_circles():
    # deeper mlp with relu for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.initializers import RandomUniform
    from matplotlib import pyplot
    # generate 2d classification dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    # define model
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


def deep_mlp_circles_tb():
    # deeper mlp for the two circles classification problem with callback
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.initializers import RandomUniform
    from keras.callbacks import TensorBoard
    # generate 2d classification dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    # define model
    init = RandomUniform(minval=0, maxval=1)
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(5, activation='tanh', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # prepare callback
    tb = TensorBoard(histogram_freq=1, write_grads=True)
    # fit model
    model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0, callbacks=[tb])


def deep_mlp_relu_circles_tb():
    # deeper mlp with relu for the two circles classification problem with callback
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.callbacks import TensorBoard
    # generate 2d classification dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    # define model
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # prepare callback
    tb = TensorBoard(histogram_freq=1, write_grads=True)
    # fit model
    model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0, callbacks=[tb])


if __name__ == '__main__':
    # plot_circles()
    # mlp_circles()
    # deep_mlp_circles()
    # deep_mlp_circles_tb()
    deep_mlp_relu_circles()
    # deep_mlp_relu_circles_tb()
