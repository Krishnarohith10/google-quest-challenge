import re
import nltk
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate, Embedding, Dense, Dropout, LSTM, BatchNormalization

stopwords = nltk.corpus.stopwords.words('english')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train_data, y_train_data = train_data.loc[:, [1, 2, 5]], train_data.loc[:, 11:]
text_cols = x_train_data.columns

def clean_texts(text):
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in set(stopwords)]
    text = ' '.join(text)
    return text

for ind in x_train_data.index:
    for col in x_train_data.columns:
        x_train_data.loc[ind, col] = clean_texts(str(x_train_data.loc[ind, col]))
    tokenizer = Tokenizer()
    for col in x_train_data.columns:
        tokenizer.fit_on_texts(x_train_data[col])
    for col in x_train_data.columns:
        x_train_data[col] = tokenizer.texts_to_sequences(x_train_data[col])

X_train_padded = []
for col in text_cols:
    X_train_padded.append(pad_sequences(x_train_data[col], maxlen=750, padding='pre'))

def model_6_lstm(tokenizer, maxlen=750):
    embs = []
    inputs = []
    for each in text_cols:
        input_layer = Input(shape=(maxlen, ))
        emb_layer = Embedding(len(tokenizer.word_index)+1, output_dim=128)(input_layer)
        embs.append(emb_layer)
        inputs.append(input_layer)
    concat = Concatenate(axis=1)(embs)
    lstm_layer_1 = LSTM(256, return_sequences=True, dropout=0.2)(concat)
    lstm_layer_2 = LSTM(256, dropout=0.1)(lstm_layer_1)
    norm_layer = BatchNormalization()(lstm_layer_2)
    dense_layer = Dense(256, activation='relu')(norm_layer)
    droput = Dropout(0.2)(dense_layer)
    dense_1 = Dense(128, activation='relu')(droput)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output = Dense(30)(dense_2)
    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = model_6_lstm(tokenizer)

model.fit(X_train_padded, y_train_data.values, batch_size=16, epochs=2, validation_split=0.2)
