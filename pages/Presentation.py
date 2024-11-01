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


dataset_df = pd.read_csv('House_Prices/submission.csv')

st.write("### Описательная статистика цен на продажу")

st.write(dataset_df['SalePrice'].describe())

st.write("### Построение распределения цен на жилье")

plt.figure(figsize=(9, 8))
sns.histplot(dataset_df['SalePrice'], color='g', bins=100, kde=True, stat='density', alpha=0.4)
plt.title('Распределение цен на продажу')
plt.xlabel('Цена продажи')
plt.ylabel('Плотность')
st.pyplot(plt)

st.write("### Оценка важности фичей")

st.write("#### SHAP")

image_path_1 = 'pages/shap.png'

if not os.path.exists(image_path_1):
    st.error(f"Файл {image_path_1} не найден.")
else:
    st.image(image_path_1, use_column_width=True)

st.write("#### Feature importance")

image_path_2 = 'pages/feat.png'

if not os.path.exists(image_path_2):
    st.error(f"Файл {image_path_2} не найден.")
else:
    st.image(image_path_2, use_column_width=True)

st.write("#### Permutation importance")

image_path_3 = 'pages/perm.png'

if not os.path.exists(image_path_3):
    st.error(f"Файл {image_path_3} не найден.")
else:
    st.image(image_path_3, use_column_width=True)


st.write("### Метрики разных моделей")

image_path = 'pages/kaggle.png'

if not os.path.exists(image_path):
    st.error(f"Файл {image_path} не найден.")
else:
    st.image(image_path, use_column_width=True)




st.header("Таблица метрик")


METRIKS = pd.DataFrame(data={'RMLSE': [0.13556, 0.15703, 0.15459, 0.13749], 
                             'RMSE': [32171.60, 31111.02, 33994.14, 31111.03], 
                             'R2':[0.8473, 0.7916, 0.8295, 0.8394]}, 
                             index=['LGBM', 'RandomForest', 'XGB', 'CatBoost'])

st.write(METRIKS)