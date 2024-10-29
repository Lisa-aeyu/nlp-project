import streamlit as st

st.title('Приложение от LSTM Team')
text = """Здесь ты можешь:
- Получить классификацию отзыва на фильм (good, bad или neutral)
- Оценить степень токсичнсти сообщения"""

st.write(text)

# def page1():
#     st.title("ships")
#     st.write("Содержимое страницы 1")

# def page2():
#     st.title("Страница 2")
#     st.write("Содержимое страницы 2")

# def page3():
#     st.title("Страница 3")
#     st.write("Содержимое страницы 3")

# def main():
#     st.sidebar.title("Навигация")
#     page = st.sidebar.selectbox("Выберите страницу:", ["ships", "Страница 2", "Страница 3"])

#     if page == "ships":
#         page1()
#     elif page == "Страница 2":
#         page2()
#     elif page == "Страница 3":
#         page3()

# if __name__ == "__main__":
#     main()