import streamlit as st
import pandas as pd
from utils import cargar_datos
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
df = cargar_datos()

st.title("Modelamiento.")
st.divider()

st.header("Explicación de los modelos utilizados.")
st.write("Decidimos utilizar tres modelos para probar cual de los tres nos daba un mejor resultado: `K-Prototypes`, `K-means`, `AgglomerativeClustering`.")
st.write("Todos los modelos fueron probando SIN y CON `PCA` para evalaur ambos resultados. " \
"Se propuso utilizar PCA para reducir la dimensionalidad de las variables numéricas tras escalarlas con RobustScaler, que es más robusto a outliers, minimizando el impacto de anomalías y mejorando la calidad de los clusters. PCA capturará el 90% por ciento de la varianza, lo que reducirá el ruido, mejorará la eficiencia computacional y facilitará la identificación de patrones en los datos sin perder información relevante.")

st.subheader("K-Prototypes.")
st.write("Lo utilizamos porque tenemos datos mixtos (numéricos y categóricos), combinando K-means (para números) y K-modes (para categorías).")


st.subheader("K-Means.")
st.write("Lo utilizamos teniendo en cuenta solo las variables númericas de nuestro dataset y transformando las variables categóricas en numéricas.")


st.subheader("AgglomerativeClustering.")
st.write("Lo utilizamos porque es un algoritmo de clustering jerárquico que empieza agrupando cada punto como su propio grupo, y los va uniendo progresivamente según su cercanía, hasta formar los clusters finales. Es útil para detectar estructuras de grupo complejas, incluso con formas irregulares.")
st.divider()



