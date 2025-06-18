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



