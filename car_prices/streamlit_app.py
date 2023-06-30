import pandas as pd
import pickle
import streamlit as st
from PIL import Image

DATASET = "https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/cars.csv"
MODEL = "model_streamlit.pickle"

df = pd.read_csv(DATASET)

with open(MODEL, 'rb') as file:
    model = pickle.load(file)

df[['brand', 'name']] = df['name'].str.split(pat = ' ', n = 1 , expand= True)
df['seats'] = df['seats'].fillna(0)
df['seats'] = df['seats'].astype(int)

image = Image.open('car.jpg')
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Demo car",
    page_icon=image,

)
st.write('## Предсказание стоимости автомобиля')

st.image(image)

brand = st.sidebar.selectbox('Выберите марку', df['brand'].unique())
year = st.sidebar.slider('Выберите год',  min_value = 1983, max_value = 2020, step=1)
km_driven = st.sidebar.slider('Выберите пробег',  min_value = 1, max_value = 2360457, step=1)
fuel = st.sidebar.selectbox('Выберите тип топлива', df['fuel'].unique())
transmission = st.sidebar.selectbox('Выберите трансмиссию', df['transmission'].unique())
owner = st.sidebar.selectbox('Какой по счету хозяин', df['owner'].unique())
seats = st.sidebar.slider('Выберите количество мест',  min_value = 0, max_value = 14, step=1)
btn_predict = st.sidebar.button('Предсказать')

#age = st.sidebar.slider("Возраст", min_value=1, max_value=80, value=20, step=1)

if btn_predict:
    X = [brand, year, km_driven, fuel, transmission, owner, seats]
    predict = model.predict(X)
    if predict > 0:
        st.write(f"## Цена вашего автомобиля {predict.round(0)} §")
    else:
        st.write('Ей место в музее')