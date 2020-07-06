import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np

tfds.disable_progress_bar()

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k', split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          as_supervised=True, with_info=True)

encoder = info.features['text'].encoder
print("Vocabulary size: ", str(encoder.vocab_size))

sample_string = "Hello Raafe"

encoded_string = encoder.encode(sample_string)
print("Encoded string is: {}".format(encoded_string))

original_string = encoder.decode(encoded_string)
print("Original String is: {}".format(original_string))

assert original_string == sample_string


