# import json
# import random
# import joblib
# import nltk
# import string
# import numpy as np
# import pickle
# import tensorflow as tf
# from nltk.stem import WordNetLemmatizer
# from tensorflow import keras
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# global responses, lemmatizer, tokenizer, le, model, input_shape
# input_shape = 10

# # import dataset answer
# def load_response():
#     global responses
#     responses = {}
#     with open('dataset/dedecorins.json') as content:
#         data = json.load(content)
#     for intent in data['intents']:
#         responses[intent['tag']]=intent['responses']

# # import model dan download nltk file
# def preparation():
#     load_response()
#     global lemmatizer, tokenizer, le, model
#     tokenizer = joblib.load(open("model/tokenizer.joblib","rb"))
#     le = joblib.load(open("model/label_encoder.joblib","rb"))
#     model = keras.models.load_model('model/chat_model.h5')
#     lemmatizer = WordNetLemmatizer()
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     nltk.download('omw-1.4', quiet=True)

# # hapus tanda baca
# def remove_punctuation(text):
#     texts_p = []
#     text = [letters.lower() for letters in text if letters not in string.punctuation]
#     text = ''.join(text)
#     texts_p.append(text)
#     return texts_p

# # mengubah text menjadi vector
# def vectorization(texts_p):
#     vector = tokenizer.texts_to_sequences(texts_p)
#     vector = np.array(vector).reshape(-1)
#     vector = pad_sequences([vector], input_shape)
#     return vector

# # klasifikasi pertanyaan user
# def predict(vector):
#     output = model.predict(vector)
#     output = output.argmax()
#     response_tag = le.inverse_transform([output])[0]
#     return response_tag

# # menghasilkan jawaban berdasarkan pertanyaan user
# def generate_response(text):
#     texts_p = remove_punctuation(text)
#     vector = vectorization(texts_p)
#     response_tag = predict(vector)
#     answer = random.choice(responses[response_tag])
#     return answer

import json
import joblib
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing

input_shape = 9
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon

words = [] # Data kata 
documents = [] # Data Kalimat Dokumen
classes = [] # Data Kelas atau Tag
intents = json.loads(open('dataset\dedecorins.json').read())
ignore_words = ['?', '!'] 

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('dedecorins.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
load_response()

# print("Responses dari variable awal nih bre:", responses)
le = joblib.load("model/label_encoder.joblib")
tokenizer = joblib.load("model/tokenizer.joblib")
lemmatizer = joblib.load("model/lemmatizer.joblib")
model = keras.models.load_model('model/chat_model.h5')


def generate_response(prediction_input):
    texts_p = []
# Menghapus punktuasi dan konversi ke huruf kecil
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)

    # Tokenisasi dan Padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    # Mendapatkan hasil keluaran pada model 
    output = model.predict(prediction_input)
    output = output.argmax()

    # Menemukan respon sesuai data tag dan memainkan voice bot
    response_tag = le.inverse_transform([output])[0]
    return random.choice(responses[response_tag])

# # import model dan download nltk file
def preparation():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


#