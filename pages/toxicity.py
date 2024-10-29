import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny-toxicity"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

def text2toxicity(text, aggregate=True):
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba


st.title("Оценка токсичности текста")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Вот данные из вашего файла:")
    st.write(data[['comment']])
    target = st.selectbox("Выберете столбец c текстами для оценки токсичности:", data.columns)

    if st.button("Оценить токсичность"):
        predictions = []
        processing_times = []
        model.eval()  # Переключение модели в режим оценки

        for text in data[target]:
            start_time = time.time()

            toxicity_score = text2toxicity(text, True)
            predictions.append(toxicity_score)

            end_time = time.time()  # Запоминаем время окончания
            processing_times.append(end_time - start_time)  # Считаем время обработки

        # Добавление результатов в DataFrame
        data['toxicity_score'] = predictions
        data['processing_time'] = processing_times  # Время обработки для каждого комментария

        # Вывод результатов
        st.write("Оценки токсичности:")
        st.write(data[['comment', 'toxicity_score', 'processing_time']].head())

        # Сохранение результатов в новый CSV файл
        output_file = "toxicities_with_scores.csv"
        data.to_csv(output_file, index=False)
        with open(output_file, 'rb') as f:
            st.download_button(
                label="Скачать файл с результатами",
                data=f,
                file_name=output_file,
                mime='text/csv'
            )

input_text = st.text_area("Или введите текст для оценки токсичности:")
if st.button("Оценить токсичность", key = '1'):
    if input_text:
        predictions = []
        processing_times = []
        model.eval()  # Переключение модели в режим оценки

        start_time = time.time()
        toxicity_score = text2toxicity(input_text, True)

        end_time = time.time()  # Запоминаем время окончания
        processing_time = end_time - start_time  # Считаем время обработки

        # Вывод результата
        st.write(f"Степень токсичности: {toxicity_score:.4f}")
        st.write(f"Время обработки: {processing_time:.4f} секунд")
    else:
        st.error("Пожалуйста, введите текст для анализа.")

text = '''


'''
st.write(text)
st.subheader('Сервис реализован с использованием  модели rubert-tiny-toxicity')
st.image('images/toxicity_metrics/toxicity_metrics.png', use_column_width=True, caption="Метрики модели на тестовом датасете")