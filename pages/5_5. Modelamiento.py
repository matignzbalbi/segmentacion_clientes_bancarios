import streamlit as st
import pandas as pd
from utils import cargar_datos
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
df = cargar_datos()

st.markdown("<h1 style='color:#00BFFF;'>Modelamiento.</h1>", unsafe_allow_html=True)
st.divider()

st.header("Introducción al Modelado.")
st.markdown("Cómo se explico brevemente en la introducción el objetivo del modelado es agrupar a los clientes en diferentes grupos de manera que sea posible proponer \
    útiles para el negocio.")
st.markdown("Con esto en mente, útilizamos diferentes modelos **no supervisados** e implementamos técnicas de reducción de dimensionalidad para evaluar su rendimiento.")

st.divider()


col1, col2 = st.columns([3, 2.4])
with col1:
    
    # Alineado (No borrar)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # Alineado (No borrar)


    st.header("Análisis de Componentes Principales (PCA)")
    st.markdown('''
            Para cada uno de los modelos se evaluaron sus resultados con PCA cómo sin este y el número de componentes principales se determino en base a las pruebas y la \
    varianza acumulada explicada.
    * **Número de componentes**: ``6``
    * **Varianza acumulada explicada**: ``~80%``
    
    Se realizaron pruebas con diferentes números de componentes y al subirlos se descubrió que el índice silueta caía significativamente, por lo cuál se decidió que 6
    es un número aceptable ya que mantiene una buena parte de la información mientras que el índice silueta no muestra un deterioro tan marcado.
    ''')

with col2:
    st.image("https://i.imgur.com/1IQtV2w.png", use_container_width=True, caption="Varianza Acumulada")



st.divider()
st.header("Algoritmos de Clustering Aplicados.")
st.markdown('''Cada uno de los modelos aplicados fue evaluado principalmente mediante el índice de silueta (ver **6. Evaluación del Modelo**) y utilizando un número de 4 clusters.<br>
Esta cantidad fue determinada mediante el método del codo (**Elbow Method**), considerando la naturaleza del problema abordado.

Por otro lado, logramos observar una alta **colinealidad** en ciertas variables por lo cuál se seleccionaron las columnas con una colinealidad baja para lograr mejores resultados en el modelamiento.
''', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

st.subheader("Modelos Aplicados.")

for modelo, proceso in [
    ("K-Prototypes.", 
     """**K-Prototypes** nos permite trabajar con variables categóricas y numéricas, por lo cuál es una opción ideal para nuestros datos.  
Realizamos un escalado de las variables numéricas junto con el filtrado de columnas para reducir la colinealidad."""
    ),
    ("K-Means.", 
     """**K-Means** solo nos permite trabajar con variables numéricas, por lo cuál utilizamos **OrdinalEncoder** para ``Education`` y **LabelEncoder** para ``Marital_Status``, nuestras dos variables categóricas."""
    ),
    ("Agglomerative.", 
     """Al igual que K-Means, el **agrupamiento aglomerativo** solo trata con variables numéricas, por lo que recibió los datos de la misma forma.  
En cuanto al método de enlace, el que mejor resultado nos proporcionó fue ``ward``."""
    ),
]:
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"**Modelo:** {modelo}")
    with c2:
        st.markdown(proceso)

    
    
    
    
