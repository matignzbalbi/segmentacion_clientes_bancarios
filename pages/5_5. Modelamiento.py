import streamlit as st
import pandas as pd
from utils import cargar_datos
from utils import escalado
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
df = cargar_datos()

st.title("Modelamiento.")
st.divider()

st.header("Escalado de los datos.")
st.write("""Antes de implementar el modelo escalamos los datos con RobustScaler debido a la alta presencia de outliers en el dataset.\
    
    Este escalado lo realizamos excluyendo las variables **binarias.**""")

escalado_code = """
ct = ColumnTransformer([
    ("num", RobustScaler(), numericas),
    ("cat", "passthrough", categoricas),
    ("bin", "passthrough", binarias)
])
"""
st.code(escalado_code, language="python")
st.divider()

st.header("K-Prototypes como modelo seleccionado.")
st.write("Debido a la naturaleza de los datos, con columnas numéricas, categóricas y binarias, utilizamos `K-Prototypes` como modelo para\
    conseguir los diferentes clústers.")

model_code = '''
kprototype = KPrototypes(n_jobs=-1, n_clusters=4, init='Huang', random_state=0)
clusters = kprototype.fit_predict(dfMatrix, categorical=catColumnsPos))
'''
st.code(model_code, language="python")

st.subheader("Número de clusters.")
st.write("El número de clusters se determino mediante el **Elbow Method**, considerando el objetivo del proyecto.")


st.header("Implementación de PCA")
st.write("""
Decidimos implementar PCA para evaluar los resultados y compararlos contra el modelo sin este tratamiento.         
Se seleccionaron **10 componentes** mediante el criterio de varianza acumulada:
- Varianza explicada acumulada: 90.2%  
- Umbral establecido: 90%  
- Reducción dimensional: de X features a 10
""")



