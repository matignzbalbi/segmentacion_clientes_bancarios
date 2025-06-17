import pandas as pd
import streamlit as st
import numpy as np
from utils import cargar_datos
from utils import limpiar_datos
from utils import features
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")


df = cargar_datos()

st.title("Análisis Exploratorio de Datos.")
st.divider()

st.header("Nulos y duplicados.")
st.subheader("Nulos.")
st.write("Analizaremos si nuestra base de datos cuenta con valores nulos y/o con valores duplicados.")
codigo = '''def dataframe_info(data):
    info = pd.DataFrame({
        "Columnas": df.columns,
        "Valores nulos": df.isnull().sum(),
        "% de nulos": round(df.isnull().mean() * 100, 2)
    })
    return info'''
st.code(codigo) 
def dataframe_info(df):
    info = pd.DataFrame({
        "Columnas": df.columns,
        "Valores nulos": df.isnull().sum(),
        "% de nulos": round(df.isnull().mean() * 100, 2)
    })
    return info
st.dataframe(dataframe_info(df))

st.write("Vemos que hay 24 valores nulos en la columna `Income`.Optamos por eliminar estos 24 registros nulos.")
codigo = '''data = data.dropna()'''
st.code(codigo)

st.subheader("Duplicados.")
st.write("Revisamos y vemos que no hay duplicados.")
codigo = '''duplicados = data[data.duplicated()]
duplicados'''
st.code(codigo)
duplicados = df[df.duplicated()]
st.dataframe(duplicados)


st.divider()
st.header("Análisis de las variables.")
st.subheader("Cambio de tipo en las variables.")
st.write("Cambiamos el type de DT_Customer a datatime y revisamos si queda alguna fecha con formato inválido.")
codigo = '''data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
fechas_invalidas = data[data['Dt_Customer'].isna()]

print("Filas con fechas inválidas (NaT):")
fechas_invalidas'''
st.code(codigo)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
fechas_invalidas = df[df['Dt_Customer'].isna()]
st.write("Observamos que no hay fechas inválidas.")
st.dataframe(fechas_invalidas)

st.subheader("Eliminación y transformación de variables.")
st.write("Analizamos los valores y frecuencias de las columnas categoricas.")
codigo = '''df = pd.DataFrame(data)

frecuencia = df['Marital_Status'].value_counts()

frecuencia'''
st.code(codigo)
frecuencia = df['Marital_Status'].value_counts()
st.dataframe(frecuencia)

st.write("También podemos ver la variable `Marital_Status` graficada en una gráfico de barras:")

conteo = df["Marital_Status"].value_counts().reset_index()
conteo.columns = ["Estado Civil", "Cantidad"]

fig = px.bar(
    conteo,
    x="Estado Civil",
    y="Cantidad",
    height=600,
    title="Distribución del Estado Civil",
)
st.plotly_chart(fig, use_container_width=True)

st.write("Dentro de esta columna encontramos las categorías `YOLO` y `Absurd`, las cuales tienen unas pocas ocurrencias y sumado a que no tienen sentido, decidimos eliminarlas.")
st.write("Por otra parte, el objetivo de este análisis de agrupar a los clientes, por lo que decidimos reducir la complejidad realizando la siguiente operación:")
st.write("Unificando Married y Together, y catalogando a todas las categorias que impliquen estar soltero en Single.")
code_ms = '''
mapeo_marital_status = {
    "Married": "Married",
    "Together": "Married",
    "Single": "Single",
    "Divorced": "Single",
    "Widow": "Single",
    "Alone": "Single",
    "Absurd": np.nan,
    "YOLO": np.nan
        }
        
df["Marital_Status"] = df["Marital_Status"].map(mapeo_marital_status)
            '''
st.code(code_ms, language="python")

st.write("De la misma forma, reducimos la complejidad para la variable **Education**, unificando PhD, Master y 2n Cycle como **PostGraduate**.")
code_ed = '''
mapeo_education = {
    'PhD': 'Postgraduate',
    'Master': 'Postgraduate',
    '2n Cycle': 'Postgraduate',
    'Graduation': 'Graduate',
    'Basic': 'Undergraduate'
}

df['Education'] = df['Education'].map(mapeo_education)
'''
st.code(code_ed, language="python")

st.write("Las columnas `Z_Revenue` y `Z_CostContact` poseen un único valor cada una\
    , por lo que no aportan información útil y decidimos eliminarlas del dataset.")

codigo = '''df = df.drop(columns=['Z_Revenue','Z_CostContact','ID'], axis=1)'''
st.code(codigo)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Z_Revenue")
    st.bar_chart(df["Z_Revenue"].value_counts(), y_label="Frecuencia", x_label="Valores")
    
with col2:
    st.subheader("Z_CostContact")
    st.bar_chart(df["Z_CostContact"].value_counts(), y_label="Frecuencia", x_label="Valores")


st.divider()

st.subheader("Creación de nuevas variables.")
df = limpiar_datos(df)
df = features(df)

st.markdown('''
            Una vez fueron limpiados, implementamos nuevas features como:
            * `Age`: Representa la edad actual del cliente.
            * `Spent`: Representa el total gastado en productos.
            * `Children`: Representa el número de hijos.
            * `Family_Size`: Representa el tamaño total de la familia.
            * `TotalAcceptedCmp`: Representa el número total de campañas aceptadas.
            * `NumTotalPurchases`: Representa el número total de compras realizadas.
            * `Customer_Tenure`: Representa el tiempo desde la último compra del cliente.
            ''')

st.dataframe(df.head(5))
st.divider()

st.subheader("Tratamiento de Outliers.")
st.write("Como primer paso, creamos un boxplot para cada una de las variables para tener un paneo general de los outliers. (No fueron graficados en Streamlit por obvias razones pero pueden verse en el código.)")
codigo = '''# Filtrar solo columnas numéricas
numeric_df = data.select_dtypes(include=np.number)

# Excluir columnas binarias (con exactamente 2 valores únicos)
non_binary_columns = [col for col in numeric_df.columns if numeric_df[col].nunique() > 2]

# Número de gráficos
num_plots = len(non_binary_columns)
cols = 2
rows = (num_plots + 1) // cols

# Crear subplots
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
axes = axes.flatten()

# Crear un boxplot para cada variable
for i, col in enumerate(non_binary_columns):
    data.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')

# Eliminar ejes vacíos
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()'''
st.code(codigo)

st.write("Para el tratamiento de los outliers creemos que la mejor opción es utilizar" \
" los tres métodos vistos en clase: IQR, Z-score, LOF y hacer un comparación entre ellos.")

codigo = '''data = df
num_vars = df.select_dtypes(include=np.number).columns.tolist()

iqr_flags = pd.DataFrame(index=data.index)

for col in num_vars:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_flags[col] = (data[col] < lower) | (data[col] > upper)

data['is_outlier_IQR'] = iqr_flags.any(axis=1)

z_scores = np.abs(stats.zscore(data[num_vars]))
data['is_outlier_Z'] = (z_scores > 3).any(axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[num_vars])
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
data['is_outlier_LOF'] = lof.fit_predict(X_scaled) == -1'''
st.code(codigo)


data = df
num_vars = df.select_dtypes(include=np.number).columns.tolist()

iqr_flags = pd.DataFrame(index=data.index)

for col in num_vars:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    iqr_flags[col] = (data[col] < lower) | (data[col] > upper)

data['is_outlier_IQR'] = iqr_flags.any(axis=1)

z_scores = np.abs(stats.zscore(data[num_vars]))
data['is_outlier_Z'] = (z_scores > 3).any(axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[num_vars])
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
data['is_outlier_LOF'] = lof.fit_predict(X_scaled) == -1

st.write("Observamos la cantidad de Outliers identificados por IQR:")
codigo = '''print("Outliers por IQR:", data['is_outlier_IQR'].sum())'''
st.code(codigo)
st.write("Outliers por IQR:", data['is_outlier_IQR'].sum())

st.write("Observamos la cantidad de Outliers identificados por Z-score:")
codigo = '''print("Outliers por Z-score:", data['is_outlier_Z'].sum())'''
st.code(codigo)
st.write("Outliers por Z-score:", data['is_outlier_Z'].sum())

st.write("Observamos la cantidad de Outliers identificados por LOF:")
codigo = '''print("Outliers por LOF:", data['is_outlier_LOF'].sum())'''
st.code(codigo)
st.write("Outliers por LOF:", data['is_outlier_LOF'].sum())


st.write("Observamos la cantidad de Outliers identificados por los TRES métodos:")
codigo = '''data['outlier_todos'] = data['is_outlier_IQR'] & data['is_outlier_Z'] & data['is_outlier_LOF']
print("Outliers detectados por los 3 métodos:", data['outlier_todos'].sum())'''
st.code(codigo)
data['outlier_todos'] = data['is_outlier_IQR'] & data['is_outlier_Z'] & data['is_outlier_LOF']
st.write("Outliers presentes en los tres métodos:", data['outlier_todos'].sum())

st.write("Debido al tamaño de nuestro dataset, considerando que es un dataset chico, decidimos imputar los Outliers con la mediana, en lugar de eliminarlos, ya que representan alrededor del 3 por ciento de nuestros datos.")
codigo = '''def replace_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = df[column].median()

    df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df

for col in num_cols:
    data = replace_outliers_with_median(data, col)'''
st.code(codigo)

