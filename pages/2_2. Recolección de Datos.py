import streamlit as st
import io
from utils import cargar_datos
import pandas as pd

st.set_page_config(layout="wide")

st.markdown("<h1 style='color:#00BFFF;'>Recolección de datos.</h1>", unsafe_allow_html=True)
st.title("Recolección de datos.")

st.divider()

st.header("Clientes bancarios.csv")
st.write("El dataset con el que trabajamos fue entregado por los profesores, por lo tanto no tuvimos que utilizar ningún tipo de herramienta o método. Más específicamente es un archivo CSV cuyas variables son tanto categóricas como numéricas, y es por eso que debimos realizar distintos tipos de mecanismos a la hora de poder trabajar con estas variables. Tiene datos demográficos como (Year_Birth), (Education), (Marital_Status), (Income). Datos de consumo como (MntWines),(MntFruits),(MntMeatProducts)" \
", y hábitos de compra (NumWebPurchases), (NumStorePurchases).")
st.divider()

st.subheader("Ir a la base de datos.")
st.markdown("Se puede ir a la base de datos clickeando en el siguiente [link](https://drive.google.com/file/d/1n2jyv8wbybziDzJoKRiM7ybXtALjs216/view?usp=drive_link)")
st.divider()

st.subheader("Cargamos la base de datos.")
codigo_cargabasededatos = '''# elegir la carpeta donde estan las bases de datos:
ruta_base = "/content/drive/My Drive"
ruta_laboratorio = os.path.join(ruta_base, "Laboratorio de Ciencia de Datos")
ruta_bases_datos = os.path.join(ruta_laboratorio, "Bases de Datos")

#elegir la base de datos que quiero usar:
ruta_archivo = os.path.join(ruta_bases_datos, "Grupo 2 - Clientes Bancarios.csv")'''
st.code(codigo_cargabasededatos,language="python")
st.divider()

st.subheader("Revisamos la base.")
codigo_cargabasededatos = '''# cargar la versión CSV:
data = pd.read_csv(ruta_archivo, sep='\t', encoding='utf-8')
data.head(5)'''
st.code(codigo_cargabasededatos,language="python")
df = cargar_datos()
st.dataframe(df.head(5))

def dataframe_info(df):
    info = pd.DataFrame({
        "Columnas": df.columns,
        "Tipo de dato": df.dtypes,
        "Valores no nulos": df.notnull().sum(),
        "Valores nulos": df.isnull().sum(),
        "% de nulos": round(df.isnull().mean() * 100, 2)
    })
    return info

st.subheader("Resumen del DataFrame")
st.dataframe(dataframe_info(df))
st.divider()
