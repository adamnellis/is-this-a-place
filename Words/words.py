
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


def words(file_path):
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from matplotlib import pyplot

    # get word list
    word_list, alphabet = read_in_words(file_path)
    max_word_length = max(len(word) for word in word_list)
    alphabet_length = len(alphabet)
    print('{0} words'.format(len(word_list)))
    print('alphabet size: {0}'.format(alphabet_length))
    print('longest word: {0} letters'.format(max_word_length))

    # split into train and test sets (every other word)
    training_words = word_list[::2]
    test_words = word_list[1::2]

    # define model
    model = Sequential()
    model.add(Dense(max_word_length, input_dim=max_word_length, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(max_word_length, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(max_word_length, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(max_word_length, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    # model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
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


if __name__ == '__main__':
    words('test-gb-places.txt')
