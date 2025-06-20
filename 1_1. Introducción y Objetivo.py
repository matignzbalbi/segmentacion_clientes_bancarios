import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


st.markdown("<h1 style='color:#00BFFF;'>Trabajo Final - Laboratorio de Ciencia de Datos</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='color:#ffffff;'>Grupo 2 – Clientes bancarios.</h2>", unsafe_allow_html=True)


st.title("Trabajo Final - Laboratorio de Ciencia de Datos.")
st.header("Grupo 2 – Clientes bancarios.")
st.divider()

st.subheader("Contexto.")
st.markdown("### 📌 Contexto")
st.write("Una institución bancaria quiere entender mejor el comportamiento de sus clientes para ofrecer\
         servicios personalizados y mejorar la retención.")

st.subheader("Objetivo.")
st.markdown("### 🎯 Objetivo")
st.write("Analizar el comportamiento financiero de los clientes y proponer una\
         forma de agruparlos en perfiles útiles para el negocio. De esta forma\
         podemos decir que estaremos trabajando en un modelo NO supervisado, específicamente en un modelo de ‘Clustering’.")

st.divider()
st.markdown("Esta aplicación se presenta a modo de resumen. El desarrollo completo del análisis se puede encontrar haciendo click [acá](https://colab.research.google.com/drive/1wo_o871ordBlu_YGJjtdV8qPToMFdtuc?usp=sharing#scrollTo=wEJysXCSHAR0).")
st.divider()

st.markdown("Alumnos: **Julian Livolsi, Tomas Martin, Matias Gonzalez Balbi, Tomas Agustin Coll**.")

cols = st.columns(2)
cols[0].markdown("**Julian Livolsi**")
cols[1].markdown("**Matias Gonzalez Balbi**")
cols[2].markdown("**Tomas Martín**")
cols[3].markdown("**Tomas Agustin Coll**")