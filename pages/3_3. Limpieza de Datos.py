import pandas as pd
import streamlit as st
from utils import cargar_datos
from utils import limpiar_datos
from utils import features
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout="wide")


df = cargar_datos()

st.title("Análisis Exploratorio de Datos.")
st.divider()

st.header("Nulos y duplicados.")
st.write("Analizaremos si nuestra base de datos cuenta con valores nulos y/o con valores duplicados.")
codigo = '''def dataframe_info(df):
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

st.write("Optamos por eliminar estos 24 registros nulos.")
codigo = '''data = data.dropna()'''
st.code(codigo)

st.write("Revisamos y vemos que no hay duplicados.")
codigo = '''duplicados = data[data.duplicated()]
duplicados'''
st.code(codigo)
duplicados = df[df.duplicated()]
st.dataframe(duplicados)


st.divider()
st.header("Análisis de las variables")
st.write("Dentro de la columna `Marital_Status` encontramos:")

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

st.write("Dentro de esta columna encontramos las categorías `YOLO` y `Absurd`, las cuales solo tienen unas pocas ocurrencias, por ende decidimos eliminarlas.")
st.write("Por otra parte, el objetivo de este análisis de agrupar a los clientes, por lo que decidimos reducir la complejidad realizando la siguiente operación:")
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

st.write("De la misma forma, reducimos la complejidad para la variable **Education**.")
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

st.write("Las columnas `Z_Revenue` y `Z_CostContact` tienen un solo valor cada una\
    , por lo que no aportan información útil y las eliminaremos de los datos.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Z_Revenue")
    st.bar_chart(df["Z_Revenue"].value_counts(), y_label="Frecuencia", x_label="Valores")
    
with col2:
    st.subheader("Z_CostContact")
    st.bar_chart(df["Z_CostContact"].value_counts(), y_label="Frecuencia", x_label="Valores")
    
df = limpiar_datos(df)
df = features(df)

st.divider()
st.header("Features")
st.markdown('''
            Una vez fueron limpiados, implementamos nuevas features como:
            * `Spent`: Representa el total gastado en productos.
            * `N_Children`: Representa el número de hijos.
            * `NumTotalPurchases`: Representa el número total de compras realizadas.
            ''')

st.dataframe(df)
