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




st.write("""# Ислледование: Цены на жилье - Передовые методы регрессии""")

st.write("""## Цель:
        Предсказать цену продажи каждого дома.
    Для каждого идентификатора в тестовом наборе вы должны предсказать значение переменной SalePrice""")


st.write("""##  Метрика:
        Оценивается по среднеквадратичнойошибке (RMSE)
    между логарифмом прогнозируемого значения и логарифмом наблюдаемой цены продажи""")


file = st.sidebar.file_uploader("Загрузите CSV-файл", type="csv")
if file is not None:
    test = pd.read_csv(file)
    st.write("Необработанные данные")
    st.write(test.head(5))
else:
    st.stop()


data = pd.read_csv("/home/zemfira/Фаза 1/Проект Х/house-prices-advanced-regression-techniques/House_Prices/train.csv")
pd.set_option('display.max_columns', None)

X, y = data.drop('SalePrice', axis=1), data['SalePrice']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)


drop_features = ['Fence', 'Alley', 'MiscFeature', 'MiscVal', 'PoolQC', 'Id']
drop_features_1 = ['BsmtFinType2', 'Street', 'BsmtHalfBath', 'LowQualFinSF', '3SsnPorch', 'Heating', 'Condition2', 'Utilities']


cat_imputing_NA = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
cat_imputing_NA_garagecond = ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType' ]

imputer_1 = ColumnTransformer(
    transformers = [
        ('num_imputer', SimpleImputer(strategy='median'), ['GarageYrBlt', 'LotFrontage']),
        ('zero_imputer', SimpleImputer(strategy='constant', fill_value=0), ['MasVnrArea']),
        ('cat_imputer_1', SimpleImputer(strategy='constant', fill_value="No Basement"), cat_imputing_NA),
        ('cat_imputer_2', SimpleImputer(strategy='most_frequent'), ['Electrical']),
        ('cat_imputer_3', SimpleImputer(strategy='constant', fill_value="No Garage"), cat_imputing_NA_garagecond),
        ('cat_imputer_4', SimpleImputer(strategy='constant', fill_value="None"), ['MasVnrType', 'MiscFeature']),
        ('cat_imputer_5', SimpleImputer(strategy='constant', fill_value="No Fireplace"), ['FireplaceQu']),
        ('cat_imputer_6', SimpleImputer(strategy='constant', fill_value="No"), ['Fence', 'Alley', 'PoolQC'])

    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)

imputed1_x_train = imputer_1.fit_transform(X_train)
imputed1_x_valid = imputer_1.transform(X_valid)

num_x_train = imputed1_x_train.select_dtypes(exclude='object')
cat_x_train = X_train.select_dtypes(include='object')

num_x_train_columns = np.array(num_x_train.columns)
minmax_scaler_columns = np.array(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'])
standard_scaler_columns = np.setdiff1d(num_x_train_columns, minmax_scaler_columns)
scaler_1 = ColumnTransformer(
    [
        ('min_max_scalling_columns', MinMaxScaler(), minmax_scaler_columns),
        ('standard_scalling_columns', StandardScaler(), standard_scaler_columns)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)

scaled1_x_train = scaler_1.fit_transform(imputed1_x_train)
scaled1_x_valid = scaler_1.transform(imputed1_x_valid)


filter_enc = np.array(['Street', 'Utilities', 'CentralAir'])
all_col_x_train_cat = np.array(cat_x_train.columns)
ordinal_enc = ['Street', 'Utilities', 'CentralAir']
target_enc = np.setdiff1d(all_col_x_train_cat, filter_enc)


encoder_1 = ColumnTransformer(
    [
        ('ordinal_encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_enc),
        ('target_encoding', TargetEncoder(), target_enc)
    ], 
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)

encoded1_x_train = encoder_1.fit_transform(scaled1_x_train, y_train)
encoded1_x_valid = encoder_1.transform(scaled1_x_valid)

col_dropped = np.array(['Fence', 'Alley', 'PoolQC'])
columns_for_scaler2 = np.setdiff1d(all_col_x_train_cat, col_dropped)
scaler_2 = ColumnTransformer(
    [
        ('standard_scalling2_columns', StandardScaler(), columns_for_scaler2)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)

scal2_x_train = scaler_2.fit_transform(encoded1_x_train, y_train)
scal2_x_valid = scaler_2.transform(encoded1_x_valid)

imputer_2 = ColumnTransformer(
    transformers = [
        ('drop_features', 'drop', drop_features),
        ('drop_features_1', 'drop', drop_features_1)

    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)

imputed2_x_train = imputer_2.fit_transform(X_train)
imputed2_x_valid = imputer_2.transform(X_valid)

cb = CatBoostRegressor()

best_preprocessor = Pipeline(
    [
        ('imputer1', imputer_1),
        ('scaler', scaler_1),
        ('encoder', encoder_1),
        ('scaler2', scaler_2),
        ('imputer2', imputer_2),
        ('model', cb)
    ]
)


best_preprocessor.fit(X_train, np.log(y_train))

y_preds = best_preprocessor.predict(X_valid)

RMLSE = np.sqrt(np.mean((np.log(y_valid) - y_preds) ** 2))
RMLSE_KAGGLE = 0,12495

R2 = 1 - (((y_valid - np.exp(y_preds))**2).sum() / ((y_valid - y_valid.mean())**2).sum())

kaggle_df = test
kaggle_index = kaggle_df['Id']

kaggle_preds = best_preprocessor.predict(kaggle_df)
kaggle_normal = np.exp(kaggle_preds)
kaggle_pred_df = pd.DataFrame(kaggle_normal, columns=['SalePrice'])
concat_kaggle_df = pd.concat([kaggle_index, kaggle_pred_df], axis=1)

csv_data = concat_kaggle_df.to_csv(index=False).encode('utf-8')


button = st.button("Вывести результаты")
if button:
    st.write(concat_kaggle_df)
else:
    st.stop()


st.header("Ключевые метрики")

col1, col2, col3 = st.columns(3)
col1.metric("**Метрика RMLSE**",
            f"{round(RMLSE, 5)}")
col2.metric("**Метрика RMLSE-KAGGLE**",
            f"{RMLSE_KAGGLE}")
col3.metric("**Метрика R2**",
            f"{round(R2, 4)}")



button_download = st.sidebar.download_button("Сохранить резльтаты", data=csv_data, file_name='submission.csv', mime="text/csv")