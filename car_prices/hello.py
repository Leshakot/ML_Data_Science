import pandas as pd
import pickle
import streamlit as st
from model import open_data, preprocess_data, split_data, load_model_and_predict

with open('model_basic.pickle', 'rb') as f:
     model_cat = pickle.load(f)

st.title('Предсказание стоимости автомобиля на вторичном рынке.')

PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv"
df = pd.read_csv(PATH)

name = st.multiselect('Выберите модель', df['name'].unique())
year = st.multiselect('Выберите год', df['year'].unique())
km_driven = st.multiselect('Выберите пробег', df['km_driven'].unique())
transmission = st.multiselect('Выберите трансмиссию', df['transmission'].unique())
seats = st.multiselect('Выберите количество мест', df['seats'].unique())

btn_predict = st.button('Предсказать')

if btn_predict:
    predict = model_cat.predict([[name, year, km_driven, transmission, seats]])
    #print(predict)
    st.write(f'Цена: {predict}')


# def prediction(name, year, km_driven, transmission, seats):
#     prediction = model_cat.predict(
#         [[name, year, km_driven, transmission, seats]])
#     print(prediction)
#     return prediction


# this is the main function in which we define our webpage
# def main():
#
#     st.title("  УУУУ ")
#
#
#     name = st.multiselect('Выберите модель', df['name'].unique())
#     year = st.multiselect('Выберите год', df['year'].unique())
#     km_driven = st.multiselect('Выберите пробег', df['km_driven'].unique())
#     transmission = st.multiselect('Выберите трансмиссию', df['transmission'].unique())
#     seats = st.multiselect('Выберите количество мест', df['seats'].unique())
#     result = ""
#
#     # the below line ensures that when the button called 'Predict' is clicked,
#     # the prediction function defined above is called to make the prediction
#     # and store it in the variable result
#     if st.button("Predict"):
#         result = prediction(name, year, km_driven, transmission, seats)
#     st.success( result)
#
#
# if __name__ == '__main__':
#     main()