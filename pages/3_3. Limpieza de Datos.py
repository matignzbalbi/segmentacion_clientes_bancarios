import pandas as pd
import streamlit as st
import numpy as np
from utils import cargar_datos
from utils import limpiar_datos
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")


df = cargar_datos()

st.markdown("<h1 style='color:#00BFFF;'>Análisis Exploratorio de Datos.</h1>", unsafe_allow_html=True)
st.divider()

st.header("Nulos y duplicados.")
st.subheader("Nulos.")
st.write("Analizaremos si nuestra base de datos cuenta con valores nulos y/o con valores duplicados.")
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

st.write("Dentro de esta columna encontramos las categorías `YOLO` y `Absurd`, las cuales tienen unas pocas ocurrencias y sumado a que no tienen sentido, decidimos reemplazarlas por la moda.")
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
    "Absurd": moda,
    "YOLO": moda
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

def features(data):
    #Edad actual
    data["Age"] = 2021-data["Year_Birth"]

    #Gasto total en diversos items
    data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

    #Total de menores
    data["Children"]=data["Kidhome"]+data["Teenhome"]

    #Miembros totales de la familia
    data["Family_Size"] = data["Marital_Status"].replace({"Single": 1, "Married":2, "Together":2})+ data["Children"]

    #Campañas totales aceptadas
    data['TotalAcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

    #Compras totales
    data['NumTotalPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data['NumDealsPurchases']

    #Años pertenecientes del cliente desde que se agrego a la base de datos
    data['Customer_Tenure'] = 2021 - data['Dt_Customer'].dt.year
    
    data = data.drop(columns=['Dt_Customer','Year_Birth'], axis=1)
    return data 

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


with st.expander("Boxplots de variables numéricas"):
    # Filtrar solo columnas numéricas
    numeric_df = df.select_dtypes(include=np.number)

    # Excluir columnas binarias
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

    # Mostrar en Streamlit
    plt.tight_layout()
    st.pyplot(fig)



st.write("Luego creamos dos histogramas con las variables que nosotros creemos podríamos identificar los outliers a simple vista: `Income` y `Age`")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histograma para Income
sns.histplot(df['Income'], bins=30, kde=True, ax=axes[0],edgecolor='white',color='lightblue',alpha=0.7)
axes[0].set_title('Distribución de Income',color='white')
axes[0].set_xlabel('Income',color='white')
axes[0].set_ylabel('Frecuencia',color='white')

# Histograma para Age
sns.histplot(df['Age'], bins=30, kde=True, ax=axes[1],edgecolor='white')
axes[1].set_title('Distribución de Age',color='white')
axes[1].set_xlabel('Age',color='white')
axes[1].set_ylabel('Frecuencia',color='white')

plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)


st.write("Al ver que notoriamente podemos identificar los outiers, decidimos eliminarlos.")
codigo = '''#Nos quedamos con los clientes que tengan un salario < 120000
data = data[data['Income']<120000]

#Nos quedamos con los clientes que tengan < 90
data = data[data['Age']<90]'''
st.code(codigo)


st.write("Para el tratamiento de los outliers de las demas variables numéricas creemos que la mejor opción es utilizar" \
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

st.write("Analizamos cuantos outliers detecta cada método y cuantos outliers son detectados por los TRES métodosal mismo tiempo.")
# Calcular las cantidades
outliers_iqr = data['is_outlier_IQR'].sum()
outliers_z = data['is_outlier_Z'].sum()
outliers_lof = data['is_outlier_LOF'].sum()
outliers_all = (
    data['is_outlier_IQR'] & data['is_outlier_Z'] & data['is_outlier_LOF']
).sum()

# Crear tabla comparativa
outlier_summary = pd.DataFrame({
    'Método': ['IQR', 'Z-Score', 'LOF', 'Los tres métodos (intersección)'],
    'Cantidad de outliers': [outliers_iqr, outliers_z, outliers_lof, outliers_all]
})

# Mostrar la tabla
st.dataframe(outlier_summary)


st.write("Debido al tamaño de nuestro dataset, considerando que es un dataset chico, decidimos imputar los Outliers con la mediana, en lugar de eliminarlos.")
