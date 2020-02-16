import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate, Bidirectional, Embedding, Dense, Dropout, LSTM

vocab_size = 25000
embedding_dims = 128
max_len = 512
batch_size = 32
epochs = 5

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train_data = train_data[['question_title', 'question_body', 'answer']]
y_train = train_data.iloc[:, 11:].values

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if((word.isalpha()==1) and (word not in set(nltk.corpus.stopwords.words('english'))))]
    text = ' '.join(text)
    return text

corpuses = []
for col in x_train_data.columns:
    corpus = []
    for text in tqdm(x_train_data[col]):
        text = clean_text(text)
        corpus.append(text)
    corpuses.append(corpus)

tokenizer = Tokenizer(num_words = vocab_size)

x_train = []
for corpus in corpuses:
    tokenizer.fit_on_texts(corpus)
    corpus_sequence = tokenizer.texts_to_sequences(corpus)
    x_train.append(pad_sequences(corpus_sequence, maxlen = max_len))

def model():
    embs = []
    inputs = []
    for each in x_train_data.columns:
        input_layer = Input(shape=(max_len, ))
        emb_layer = Embedding(vocab_size, embedding_dims)(input_layer)
        embs.append(emb_layer)
        inputs.append(input_layer)
    concat = Concatenate(axis=1)(embs)
    lstm_1 = Bidirectional(LSTM(128, activation='relu', dropout=0.2, recurrent_dropout=0.2))(concat)
    dense_1 = Dense(64, activation='relu')(lstm_1)
    dropout_1 = Dropout(0.5)(dense_1)
    output = Dense(30, activation='sigmoid')(dropout_1)
    model = Model(inputs=inputs, outputs=[output])

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model

model = model()

model.summary()

history = model.fit(x_train, y_train, batch_size=batch_size, 
                    epochs=epochs, validation_split=0.25, shuffle=True)
'''
range_epochs = range(1, epochs+1)

plt.plot(range_epochs, history.history['accuracy'], 'r')
plt.plot(range_epochs, history.history['val_accuracy'], 'b')
plt.title('Train and Validation Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(range_epochs, history.history['loss'], 'r')
plt.plot(range_epochs, history.history['val_loss'], 'b')
plt.title('Train and Validation Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

x_test_data = test_data[['question_title', 'question_body', 'answer']]

corpuses = []
for col in x_test_data.columns:
    corpus = []
    for text in tqdm(x_test_data[col]):
        text = clean_text(text)
        corpus.append(text)
    corpuses.append(corpus)

x_test_padded = []
for corpus in corpuses:
    tokenizer.fit_on_texts(corpus)
    corpus_sequence = tokenizer.texts_to_sequences(corpus)
    x_test_padded.append(pad_sequences(corpus_sequence, maxlen = max_len, padding='post'))

prediction = model.predict(x_test_padded, batch_size=32)

submission = pd.read_csv('sample_submission.csv')

for key, col in enumerate(submission.columns):
    if col != 'qa_id':
        submission[col] = prediction[:, key-1]

submission.to_csv('submission.csv', index=False)
'''