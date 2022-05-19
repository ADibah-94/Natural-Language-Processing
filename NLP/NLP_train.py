# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:29:23 2022

@author: HP
"""

import pandas as pd
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
import json 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
import datetime
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LOG_PATH = os.path.join(os.getcwd(), 'logs') 
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

#%% EDA

# Step 1) Data loading

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

df = pd.read_csv(URL)

text = df['text']                 # x_train
text_dummy = text.copy()

category = df['category']
category_dummy = category.copy()  # y_train

# Step 2) Data Inspection

text_dummy[3]
category_dummy[3]

# Step 3) Data Cleaning

# To remove html tags (1st 4 loop)
for index, text in enumerate(text_dummy):
    text_dummy[index] = re.sub('<.*?>', ' ', text)
    
# To convert to lowercase and split it and to remove numerical text (2nd 4 loop)
for index, text in enumerate(text_dummy):
    text_dummy[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()

# Step 4) Features Selection
# Step 5) Data preprocessing
# Data vectorization

num_words = 10000
oov_token = '<OOV>'

# tokenizer to vectorize the words
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(text_dummy)

# To save the tokenizer for deployment purpose
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
token_json = tokenizer.to_json()

with open(TOKENIZER_JSON_PATH,'w') as json_file:
    json.dump(token_json, json_file)

# to observe the number of words
word_index = tokenizer.word_index
print(word_index)
print(dict(list(word_index.items())[0:10]))

# to vectorize the sequences of text
text_dummy = tokenizer.texts_to_sequences(text_dummy)

temp = [np.shape(i) for i in text_dummy]  # to check the number words inside the list

np.mean(temp) # mean of the words --> 386

text_dummy = pad_sequences(text_dummy, 
                             maxlen=200, 
                             padding='post', 
                             truncating='post')

# One-hot encoding for label

one_hot_encoder = OneHotEncoder(sparse=False)
category_encoded = one_hot_encoder.fit_transform(np.expand_dims(category_dummy,
                                                            axis=-1))
# split train test

X_train,X_test,y_train,y_test = train_test_split(text_dummy, 
                                                 category_encoded, 
                                                 test_size=0.3,
                                                 random_state=123)
X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0))

#%% Model Creation

model = Sequential()
model.add(Embedding(num_words, 64))  # added the embedding layer
model.add(Bidirectional(LSTM(32,return_sequences=True))) #added bidirectional
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))    #added bidirectional
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()

#%% Callbacks

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


#%% Compile & Model fitting

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])

hist =model.fit(X_train, y_train, 
                epochs=3, 
                validation_data=(X_test, y_test),
                callbacks=[tensorboard_callback])

#%% Model Evaluation
# Preallocation of of memory approach

predicted_advanced = np.empty([len(X_test), 5])
for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))
    

#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% Model Deployment

model.save(MODEL_SAVE_PATH)













