import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
from catboost import CatBoostRegressor
sns.set_style("darkgrid")

RANDOM_STATE = 42

DATASET_PATH = "https://raw.githubusercontent.com/evgpat/edu_stepik_from_idea_to_mvp/main/datasets/cars.csv"

## Загрузка и обзор данных

### Загрузка

# загрузка данных
df = pd.read_csv(DATASET_PATH)
# случайные три записи из датасета
df.sample(3)

df.info() # информация от столбцах

(df.isna().agg(['sum', 'mean'])
    .style.set_caption('Количество пропусков')
    .set_table_styles([{'selector': 'caption',
                     'props': [('color', 'black'), ('font-size', '15px')]
                     }]))


print('Количество уникальных значений в каждой колонке:')
for index in df.columns:
    row = df[index].nunique()
    print(f'Уникальный значений {index}: {row}')


df['fuel'].value_counts(normalize=True, ascending=False).to_frame().style.background_gradient(
    cmap='Blues').format(precision=3)

df['seller_type'].value_counts(normalize=True, ascending=False).to_frame().style.background_gradient(
    cmap='Blues').format(precision=3)


df['transmission'].value_counts(normalize=True, ascending=False).to_frame()

df['selling_price'].hist(bins=150, color='steelblue',figsize=(15, 5), ec="darkgrey")
print(f"Среднее значение: {df['selling_price'].mean().round(2)}")
print(f"Медианное значение: {df['selling_price'].median().round(2)}")
print(f"Минимальное значение: {df['selling_price'].min().round(2)}")
print(f"Максимальное значение: {df['selling_price'].max().round(2)}")
plt.title('Распределение стоимости автомобиля', fontsize=15)
plt.xlabel('Стоимость', fontsize=12)
plt.ylabel('Количество автомобилей', fontsize=10)
plt.show()

df[df['selling_price'] == df['selling_price'].max()]


df['selling_price'].hist(bins=150, range=(0, df['selling_price'].quantile(0.95)), color='steelblue',figsize=(15, 5), ec="darkgrey")
plt.title('Распределение стоимости автомобиля без выбросов', fontsize=15)
plt.xlabel('Стоимость', fontsize=12)
plt.ylabel('Количество автомобилей', fontsize=10)
plt.show()


fig = plt.figure(figsize=(9,9))

sns.heatmap(df.corr(numeric_only = True), annot=True, cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
plt.show()


df.head(3)

df[df['mileage'].isna()].iloc[:5]

df_isna = df[(df['mileage'].isna() == True)
   & (df['engine'].isna() == True)
       & (df['seats'].isna() == True)]
df_isna.isna().sum()


df[(df['engine'].isna() == False)
       & (df['torque'].isna() == True)]

df[(df['engine'] == '796 CC') & (df['fuel'] == 'CNG')]

df['mileage'] = df['mileage'].str.split(' ').str[0]

df['mileage'] = df.groupby('year')['mileage'].transform(lambda x: x.fillna(x.median()))

df[df['mileage'].isna()]

df['mileage'] = df['mileage'].astype('float')

df['mileage'] = df['mileage'].fillna(df['mileage'].quantile(0.95))

df['engine'] = df['engine'].str.split(' ').str[0]

df['max_power'] = df['max_power'].str.split(' ').str[0]

df['engine'] =df['engine'].astype(float)
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

df['engine'].corr(df['max_power'])

df['max_power'].describe()

df[df['fuel'] == 'Petrol']['max_power'].describe()

df[df['fuel'] == 'Diesel']['max_power'].describe()

df['max_power'] = df.groupby('fuel')['max_power'].transform(lambda x: x.fillna(x.median()))

df['engine'] = df.groupby('fuel')['engine'].transform(lambda x: x.fillna(x.median()))

df['seats'] = df['seats'].fillna(df['seats'].median())

cat = df

df = df.drop('torque', axis = 1)

cat['torque'] = cat['torque'].fillna('unknown')

fig = plt.figure(figsize=(9,9))

sns.heatmap(df.corr(numeric_only = True), annot=True, cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
plt.show()

## Машинное обучение

### Линейна регрессия

df_get = pd.get_dummies(df)

df_get.shape

# разделим наши данные на признаки (матрица X) и целевую переменную (y)
X = df_get.drop(columns=['selling_price'])
y = df_get['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = RANDOM_STATE)

scaler = StandardScaler()

X_train_st = scaler.fit_transform(X_train)  #обучаем scaler и одновременно трансформируем матрицу для обучающей выборки
print(X_train_st[:2])

X_test_st = scaler.transform(X_test)

model = Lasso()

model.fit(X_train_st, y_train)

predictions = model.predict(X_test_st)

## Изучим результаты

# функция выводящая метрики
def error(y_test, predictions):
    print('MSE: {:.2f}'.format(mean_squared_error(y_test, predictions)))
    print('MAE: {:.2f}'.format(mean_absolute_error(y_test, predictions)))
    print('MAPE: {:.2f}'.format(mean_absolute_percentage_error(y_test, predictions)))

error(y_test, predictions)

model.coef_, model.intercept_

comparison = pd.DataFrame({'y_test': y_test, 'predictions': predictions})

comparison['predictions'] = comparison['predictions'].round(0)

comparison['abs'] = np.abs(comparison['y_test'] - comparison['predictions'])

comparison.sort_values(by = 'abs', ascending=False).iloc[:5]

df[df['selling_price'] == 3200000].iloc[:3]

comparison['abs'].describe().apply("{0:.2f}".format)

### CatBoost

X = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats']

cat_features = ['name', 'fuel','seller_type', 'transmission', 'owner', 'torque']

y = ['selling_price']

train, test = train_test_split(cat, train_size=0.6, random_state = RANDOM_STATE)

val, test = train_test_split(test,train_size=0.5, random_state = RANDOM_STATE)

model_cat = CatBoostRegressor(cat_features=cat_features,
                              learning_rate = 0.75,
                              eval_metric = 'MAPE',
                              random_seed = RANDOM_STATE,
                              verbose=100)

model_cat.fit(train[X],train[y],eval_set=(val[X],val[y]))

test['pred'] = model_cat.predict(test[X])

error(test[y], test['pred'])

test['abs'] = np.abs(test['selling_price'] - test['pred']).round(2)

test['pred'] = test['pred'].round(0)

test[['selling_price', 'pred', 'abs']].sort_values(by = 'abs', ascending=False).iloc[:5]

test['abs'].describe().apply("{0:.2f}".format)

import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(model_cat, f)

#а так модель можно загрузить из файла:
# with open('model.pickle', 'rb') as f:
#     model_cat = pickle.load(f)


X = ['name', 'year', 'km_driven', 'fuel',
       'transmission', 'seats']

cat_features = ['name', 'transmission', 'fuel']

y = ['selling_price']

train, test = train_test_split(cat, train_size=0.6, random_state = RANDOM_STATE)

val, test = train_test_split(test,train_size=0.5, random_state = RANDOM_STATE)

model_cat_basic = CatBoostRegressor(cat_features=cat_features,
                              learning_rate = 0.75,
                              eval_metric = 'MAPE',
                              random_seed = RANDOM_STATE,
                              verbose=100)

model_cat_basic.fit(train[X],train[y],eval_set=(val[X],val[y]))

test['pred_basic'] = model_cat_basic.predict(test[X])

error(test[y], test['pred_basic'])

import pickle

with open('model_basic.pickle', 'wb') as f:
    pickle.dump(model_cat_basic, f)

#а так модель можно загрузить из файла:
# with open('model.pickle', 'rb') as f:
#     model_cat = pickle.load(f)

test['pred_basic']