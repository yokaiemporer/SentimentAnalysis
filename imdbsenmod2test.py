import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from keras.datasets import imdb 
from keras.models import Sequential 
from keras.models import Model
from keras.layers import  Input
from keras.utils import to_categorical
from keras.layers import Dense 
from keras.layers import LSTM 
from keras.layers import Flatten
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence 
from keras.layers import TimeDistributed
# # fix random seed for reproducibility 
# numpy.random.seed(7) 
import pickle
# # load the dataset but only keep the top n words, zero the rest 
top_words = 15000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print((list(X_train)[:10], list(y_train)[:10]))


y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
import keras
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
print(list(word_to_id)[:20])
print(list(word_to_id.items())[:20])
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
import numpy as np
id_to_word = {value:key for key,value in word_to_id.items()}
# print(' '.join(id_to_word[id] for id in X_train[0] ))
from numpy import array

# truncate and pad the review sequences 
max_review_length = 250 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length) 
print(pd.DataFrame(X_train).head())
# create the model 

# embedding_vector_length = 32
embedding_vector_length = 128

import tensorflow as tf 
filename = "my_model.h5"

model=tf.keras.models.load_model(filename)
scores = model.evaluate(X_test, y_test, verbose=0) 

plt.show()
print("Accuracy: %.2f%%" % (scores[1]*100))
sentences=[ "that movie is bad"
, "awesome"
,"it was very good"
,"i hate it"
,"wow",
"meh",
"its bad"
,"heroine was the worst"
,"im not sure if its ok"]
# bad="i hate it"
for i in range(10):
    print()
for review in sentences:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length) 
    val=model.predict(array([tmp_padded][0]))[0]
    idx=np.argmax(val)
    print("sentence::::")
    if idx==1:
        result="POSITIVE"
    else:
        result="NEGATIVE"
    print("%s . Sentiment is %s  and accuracy : %s" % (review,result,val))
    