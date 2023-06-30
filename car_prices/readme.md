## Прогноз стоимости подержанного автомобиля  

### Приложение 
[Carprices](https://carprices.streamlit.app/)


### Задача
Предсказать стоимость поддержанного автомобиля.   
Создать приложение в Streamlit

### Описание проекта 
Датасет содержит информацию о характеристиках и ценах подержанных автомобилей в некоторой стране.  
Целевая переменная (таргет, числовая) – `selling_price`: цена продажи

**Признаки**  
`name` (string): модель автомобиля  
`year` (numeric, int): год выпуска с завода-изготовителя  
`km_driven` (numeric, int): пробег на дату продажи  
`fuel` (categorical: Diesel или Petrol, или CNG, или LPG, или electric): тип топлива  
`seller_type` (categorical: Individual или Dealer, или Trustmark Dealer): продавец  
`transmission` (categorical: Manual или Automatic): тип трансмиссии  
`owner` (categorical: First Owner или Second Owner, или Third Owner, или Fourth & Above Owner): какой по счёту хозяин  
`mileage` (string, по смыслу числовой): пробег, требует предобработки  
`engine` (string, по смыслу числовой): рабочий объем двигателя, требует предобработки  
`max_power` (string, по смыслу числовой): пиковая мощность двигателя, требует предобработки  
`torque` (string, по смыслу числовой): крутящий момент, требует предобработки  
`seats` (numeric, float; по смыслу categorical, int)  

  
### Навыки и инструменты
Pandas, Python, NumPy, Sklearn, Matplotlib, Seaborn, CatBoost, Streamlit, предобработка данных.


![DS2](https://github.com/Leshakot/ML_Data_Science/assets/119577732/0496b737-0afc-462e-b46e-9319ec8559b2)
