import numpy as np
import pandas as pd
import re
import string
import torch
import matplotlib.pyplot as plt
import os
import torchutils as tu
import json
import sys
import streamlit as st

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAveragePrecision
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from time import time
from collections import Counter
from typing import Union
from dataclasses import dataclass
import torch.nn.functional as F

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
stop_words = set(stopwords.words('russian'))
import warnings
warnings.filterwarnings('ignore')
import time


sys.path.append('..')
from models.lamberts_funcs import data_preprocessing, l_predict_class
from models.lamberts_funcs import my_padding, train3, logs_dict_multi_metrics
from models.lstm_model import LSTMClassifier
SEQ_LEN = 111
with open("models/vocab_to_int.json", "r", encoding="utf-8") as json_file:
    vocab_to_int = json.load(json_file)


model_lstm = torch.load('models/model_lstm.pth', map_location=torch.device('cpu'))
model_lstm.eval()
st.write("""
### Модель для классификации текста на русском языке
- **Тип модели**: RNN (LSTM с механизмом attention)
- **Препроцессинг**: Выполнен вручную, с предварительной обработкой текста
- **Пэддинг**: Заполнение нулями справа, обрезание текста также справа
- **Максимальная длина строки**: 111 слов
- **Кодировка меток**: Ordinal Encoder для преобразования категорий в числовые значения
""")



txt = st.text_area(
    "Text to analyze",
)

if st.button("Предсказать"):
    start_time = time.time()
    class_label, prob = l_predict_class(
        model_lstm,
        txt,
        preprocess_fn=data_preprocessing,
        padding=my_padding,
        seq_len=SEQ_LEN,
        vocabulary=vocab_to_int,
        device='cpu'
    )
    elapsed_time = time.time() - start_time
    st.write(f"Класс: {class_label}")
    st.write(f"С вероятностью: {round(prob, 2)}")
    st.write(f"Время предсказания: {elapsed_time:.2f} секунд")
