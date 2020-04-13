import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import imdb 
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import  Input
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense 
from tensorflow.python.keras.layers import LSTM 
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers.embeddings import Embedding 
from tensorflow.python.keras.preprocessing import sequence 
from tensorflow.python.keras.layers import TimeDistributed
# # fix random seed for reproducibility 
# numpy.random.seed(7) 
import pickle
from sklearn.utils import shuffle
# # load the dataset but only keep the top n words, zero the rest 
top_words = 15000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
print
(X_train, y_train), (X_test, y_test)=shuffle(X_train, y_train),shuffle(X_test, y_test)
print((list(X_train)[:10], list(y_train)[:10]))


y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
import keras
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
# print(list(word_to_id)[:20])
# print(list(word_to_id.items())[:20])
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
embedding_vector_length = 128

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense, Dropout
from tensorflow.python.keras.layers import SpatialDropout1D
from tensorflow.python.keras.layers import Embedding
model = Sequential()
model.add(Embedding(15001, embedding_vector_length,     
                                     input_length=250) )
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', 
                           metrics=['accuracy'])
r=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64) 
# Final evaluation of the model 
import tensorflow as tf 
filename = "my_model.h5"
model.save(filename)
model=tf.keras.models.load_model(filename)
scores = model.evaluate(X_test, y_test, verbose=0) 
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
if "binary_accuracy" in r.history.keys():
	plt.plot(r.history['binary_accuracy'], label='acc')
plt.plot(r.history['accuracy'], label='acc')
 
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
print("Accuracy: %.2f%%" % (scores[1]*100))
bad = "that movie bad"
good = "awesome"
good2="it was very good"
bad2="it seemed like crap"
# bad="i hate it"
for i in range(10):
    print()
for review in [good,bad,good2,bad2]:
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
    