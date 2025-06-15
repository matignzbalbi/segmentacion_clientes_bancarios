import streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

st.title("Segmentación de Clientes Bancarios.")
st.divider()

st.subheader("Contexto.")
st.write("Una institución bancaria quiere entender mejor el comportamiento de sus clientes para ofrecer\
         servicios personalizados y mejorar la retención.")

st.subheader("Objetivo.")
st.write("Analizar el comportamiento financiero de los clientes y proponer una\
         forma de agruparlos en perfiles útiles para el negocio.")

st.divider()
st.markdown("Esta aplicación se presenta a modo de resumen. El desarrollo completo del análisis se puede encontrar haciendo click [acá](https://colab.research.google.com/drive/1wo_o871ordBlu_YGJjtdV8qPToMFdtuc?usp=sharing#scrollTo=wEJysXCSHAR0).")