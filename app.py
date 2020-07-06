import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds
# from tqdm.keras import TqdmCallback
# import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = df.load_data(num_words=85000)

# print(train_data[0])

# Word Mapping
word_index = df.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Reversing Keys and Values in the above dict
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'],
                                                        padding='post', maxlen=225)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'],
                                                       padding='post', maxlen=225)


def decode(text):
    return " ".join(reverse_word_index.get(i, '?') for i in text)


# print(decode(train_data[38]) + "\n" + str(len(train_data[1])))

'''
# ye rakha Model
model = keras.Sequential()
model.add(keras.layers.Embedding(85000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Splitting training data into 2 sets (validation set & new train_data) in order to check how well 
# our model is performing based on the tweeks on the training data in the new data
x_val = train_data[:11000]
x_train = train_data[11000:]

y_val = train_labels[:11000]
y_train = train_labels[11000:]

# training our model
fitModel = model.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val), verbose=0,
                     callbacks=[TqdmCallback(verbose=0)])

results = model.evaluate(test_data, test_labels)
print(results)                     
'''


def review_encode(rev):
    encoded = [1]
    for word in rev:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


def rev_eval(x):
    if 0 <= x < 0.25:
        print("\n\nRATING --> 1/5")
    elif 0.25 <= x < 0.45:
        print("\n\nRATING --> 2/5")
    elif 0.45 <= x < 0.75:
        print("\n\nRATING --> 3/5")
    elif 0.75 <= x < 0.90:
        print("\n\nRATING --> 4/5")
    else:
        print("\n\nRATING --> 5/5")


model = keras.models.load_model('imdb_review.h5')

with open('interstellar_trash.txt', encoding='utf-8') as file:
    for line in file.readlines():
        nline = line.replace(',', '').replace('.', '').replace('\"', '').replace('(', '').replace(')',
                                                                                                  '').strip().split(
            ' ')
        encoded_rev = review_encode(nline)
        encoded_rev = keras.preprocessing.sequence.pad_sequences([encoded_rev], value=word_index['<PAD>'],
                                                                 padding='post', maxlen=250)
        prediction = model.predict(encoded_rev)
        print("\n\nORIGINAL REVIEW :\n {}".format(line))
        print("\n\nENCODED REVIEW :\n {}".format(encoded_rev), '\n')
        rev_eval(float(prediction[0]))

'''
test_review = test_data[69]
prediction = model.predict([test_review])
print("Review: ")
print(decode(test_review))
print("Prediction: {}".format(prediction[69]))
print("Actual: {}".format(test_labels[69]))
'''
