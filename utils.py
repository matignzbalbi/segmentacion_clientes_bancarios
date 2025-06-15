import pandas as pd
import streamlit as st
import numpy as np



@st.cache_data
def cargar_datos():
    df = pd.read_csv("Grupo 2 - Clientes Bancarios.csv", sep= "\t")
    return df

def limpiar_datos(data):

    # Fechas
    
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
            
    # Mapeo Marital_Status
    
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
    
    data.loc[:,"Marital_Status"] = data["Marital_Status"].map(mapeo_marital_status)
    
    
    # Mapeo Education
    
    mapeo_education = {
    'PhD': 'Postgraduate',
    'Master': 'Postgraduate',
    '2n Cycle': 'Postgraduate',
    'Graduation': 'Graduate',
    'Basic': 'Undergraduate'
    }

    data.loc[:,'Education'] = data['Education'].map(mapeo_education)
    
    # Eliminamos columnas no útiles.
    
    data = data.drop(columns=['Z_Revenue','Z_CostContact','ID'], axis=1)
    
    data = data.dropna()

    return data

def features(data):
    #Edad actual
    data["Age"] = 2021-data["Year_Birth"]

    #Gasto total en diversos items
    data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

    #Total de menores
    data["Children"]=data["Kidhome"]+data["Teenhome"]

    #Miembros totales de la familia
    data["Family_Size"] = data["Marital_Status"].replace({"Single": 1, "Married":2, "Together":2})+ data["Children"]

    #Paternidad
    data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

    #Campañas totales aceptadas
    data['TotalAcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

    #Compras totales
    data['NumTotalPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data['NumDealsPurchases']

    #Años pertenecientes del cliente desde que se agrego a la base de datos
    data['Customer_Tenure'] = 2021 - data['Dt_Customer'].dt.year
    
    data = data.drop(columns=['Dt_Customer','Year_Birth'], axis=1)
    return data 