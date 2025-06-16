import pandas as pd
import streamlit as st
from utils import cargar_datos
from utils import limpiar_datos
from utils import features
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px



df = cargar_datos()

nulos = df.isnull().sum()
nulos.columns = ['Campo', 'Suma de nulos']
print(nulos)