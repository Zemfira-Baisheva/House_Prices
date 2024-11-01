import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import streamlit as st
import os


dataset_df = pd.read_csv('/home/zemfira/Фаза 1/Проект Х/house-prices-advanced-regression-techniques/House_Prices/submission.csv')

st.write("### Описательная статистика цен на продажу")

st.write(dataset_df['SalePrice'].describe())

st.write("### Построение распределения цен на жилье")

plt.figure(figsize=(9, 8))
sns.histplot(dataset_df['SalePrice'], color='g', bins=100, kde=True, stat='density', alpha=0.4)
plt.title('Распределение цен на продажу')
plt.xlabel('Цена продажи')
plt.ylabel('Плотность')
st.pyplot(plt)

st.write("### Метрики разных моделей")

image_path = 'pages/kaggle.png'

if not os.path.exists(image_path):
    st.error(f"Файл {image_path} не найден.")
else:
    st.image(image_path, use_column_width=True)


st.header("Таблица метрик")

col1, col2, col3 = st.columns(3)
col1.metric("**Метрика RMLSE**",
            f"{round(RMLSE, 5)}")
col2.metric("**Метрика RMLSE-KAGGLE**",
            f"{RMLSE_KAGGLE}")
col3.metric("**Метрика R2**",
            f"{round(R2, 4)}")