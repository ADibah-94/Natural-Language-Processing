# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:26:44 2022

@author: HP
"""
    
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional


class ExploratoryDataAnalysis():
    
    def __init(self):
        pass
    
    def remove_tags(self,data):
        '''


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        for index, text in enumerate(data):
            # text_dummy[index] = text.replace('<br />', '')
            data[index] = re.sub('<.*?>', '', text)
            
        return data 
    
    def lower_split(self,data):
        '''
        This function converts all letters into lowercase and split into list.
        Also filters numerical data
        
        Parameters
        ----------
        data : Array
            RAW TRAINING DATA CONTAINING STRINGS.

        Returns
        -------
        data : List
            CLEANED DATA WITH ALL LETTERS CONVERTED INTO LOWERCASE.

        '''
        
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
            
        return data
        
    def category_tokenizer(self,data,token_save_path,
                            num_words=10000,oov_token='<OOV>',prt=False):
    
        # Tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
    
        # To save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()
    
        with open(token_save_path, 'w') as json_file:
            json.dump(token_json, json_file)
        
        # to observe the number of words
        word_index = tokenizer.word_index
    
        if prt == True:
            # to view the tokenized words
            # to print(word_index)
            print(dict(list(word_index.items())[0:10]))
        
        # To vectorize the sequences of text 
        data = tokenizer.texts_to_sequences(data)
    
        return data

    def category_pad_sequence(self,data):
        return pad_sequences(data, maxlen=200,padding='post',
                             truncating='post')
        
class ModelCreation():
    
    def __init__(self):
        pass
    
    
    def lstm_layer(self,num_words, nb_categories, 
                   embedding_output=64, nodes=32, dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))  # added the embedding layer
        model.add(Bidirectional(LSTM(nodes,return_sequences=True))) #added bidirectional
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))    #added bidirectional
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model
    
    def simple_lstm_layer(self,num_words,nb_categories,
                          embedding_output=64, nodes=32,dropout=0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))  # added the embedding layer
        model.add(LSTM(nodes,return_sequences=True)) #added bidirectional
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model

#%% 
if __name__ == '__main__':
    
    import os
    import pandas as pd

    PATH_LOGS = os.path.join(os.getcwd(), 'logs')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5') 
    TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')

    URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

    df = pd.read_csv(URL)
    text = df['text']                
    category = df['category']

    #%%

    eda = ExploratoryDataAnalysis()
    test = eda.remove_tags(text)
    test = eda.lower_split(test)

    test = eda.category_tokenizer(test, token_save_path=TOKENIZER_JSON_PATH)

    test = eda.category_pad_sequence(test)

    #%%

    nb_categories = len(category.unique())
    mc = ModelCreation()
    model = mc.simple_lstm_layer(10000, nb_categories)





