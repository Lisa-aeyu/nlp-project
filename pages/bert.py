import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import torch
import os
import time
from torch import nn
import matplotlib.pyplot as plt

# импортируем трансформеры
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from torch.utils.data import DataLoader, random_split 

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import pickle
import streamlit as st

device = 'cpu'# if torch.cuda.is_available() else 'cuda'

current_dir = os.path.dirname(os.path.abspath(__file__))
lr_model_path = os.path.join(current_dir, '../models/log_reg_model_di.pkl')
le_path = os.path.join(current_dir, '../models/label_encoder_di.pkl')

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2", num_labels=3).to(device) 

# Загружаем модель логистической регрессии
with open(lr_model_path, 'rb') as file:
    lr_model = pickle.load(file)

# Загружаем LabelEncoder
with open(le_path, 'rb') as file:
    le = pickle.load(file)

# Функция для обработки текста с помощью BERT
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

# Прогнозирование класса
def predict(text):
    input_ids, attention_mask = preprocess_text(text)
    with torch.no_grad():
        outputs = AutoModel.from_pretrained("cointegrated/rubert-tiny2")(input_ids, attention_mask=attention_mask)
        # Используйте только выходы последнего слоя
        last_hidden_states = outputs.last_hidden_state[:, 0, :].numpy()  # Значение для токена [CLS]
        prediction = lr_model.predict(last_hidden_states)
        return le.inverse_transform(prediction)

# Настройка интерфейса Streamlit
st.title("Классификация текста с помощью BERT")

text_input = st.text_area("Введите текст для классификации:")

if st.button("Классифицировать"):
    if text_input:
        start_time = time.time()
        predicted_class = predict(text_input)
        elapsed_time = time.time() - start_time
        st.write(f"Предсказанный класс: {predicted_class[0]}")
        st.write(f"Время предсказания: {elapsed_time:.2f} секунд")
        # Выводим метрики
        st.write(f"F1-score: 0.53")
        st.write(f"Precision: 0.57")
        st.write(f"Recall: 0.54")
        st.write(f"Accuracy: 0.68")
    else:
        st.error("Пожалуйста, введите текст для классификации.")


