import streamlit as st
import joblib
import re
import time

# Загрузка моделей и объектов
catboost_model = joblib.load('models/catboost_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Функция предобработки текста
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)  # Убираем пунктуацию
        text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
    return text

# Функция предсказания
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    review_tfidf = vectorizer.transform([processed_review])
    
    # Получение вероятностей
    catboost_proba = catboost_model.predict_proba(review_tfidf)
    rf_proba = rf_model.predict_proba(review_tfidf)
    
    # Усреднение вероятностей
    ensemble_proba = (catboost_proba + rf_proba) / 2
    prediction_encoded = ensemble_proba.argmax(axis=1)
    
    # Декодирование предсказания
    predicted_class = label_encoder.inverse_transform(prediction_encoded)
    return predicted_class[0]

# Интерфейс Streamlit
st.title("Классификация текста с помощью TF-IDF")

review_text = st.text_input("Введите ваш отзыв:")

if st.button("Предсказать"):
    start_time = time.time()
    predicted_class = predict_sentiment(review_text)
    elapsed_time = time.time() - start_time
    st.write("Предсказанный класс:", predicted_class)
    st.write(f"Время предсказания: {elapsed_time:.2f} секунд")

# Метрики
f1_macro_catboost = 0.64  # заменить на реальные значения
f1_macro_rf = 0.63
f1_macro_ensemble = 0.67

# Таблица с метриками
st.write("## Сравнение моделей по метрике F1-macro")
st.table({
    "Модель": ["CatBoost", "Random Forest", "Ансамбль"],
    "F1-macro": [f1_macro_catboost, f1_macro_rf, f1_macro_ensemble]
})