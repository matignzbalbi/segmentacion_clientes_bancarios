import pandas as pd
import streamlit as st
import numpy as np



@st.cache_data
def cargar_datos():
    data = pd.read_csv("Grupo 2 - Clientes Bancarios.csv", sep= "\t")
    return data

def limpiar_datos(data):

    # Fechas
    
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y', errors='coerce')

    #children
    data["Children"] = data["Kidhome"] + data["Teenhome"]

    # Mapeo Marital_Status
    moda = data['Marital_Status'].mode()[0]
    mapeo_marital_status = {
        "Married": "Married",
        "Together": "Together",
        "Single": "Single",
        "Divorced": "Single",
        "Widow": "Single",
        "Alone": "Single",
        "Absurd": moda,
        "YOLO": moda
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
    
    data = data.drop(columns=['Z_Revenue','Z_CostContact','ID'], axis=1, errors = 'ignore')
    
    data = data.dropna()

    return data

def features(data):
    #Edad actual
    data["Age"] = 2025-data["Year_Birth"]

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
    data['Customer_Tenure'] = 2025 - data['Dt_Customer'].dt.year
    
    data = data.drop(columns=['Dt_Customer','Year_Birth'], axis=1)
<<<<<<< HEAD
    return data



# Modelado 
def escalado(df):
    categoricas = ["Marital_Status", "Education"]

    binarias = [col for col in df.columns if df[col].nunique() <= 2]
    binarias = [col for col in binarias if col not in categoricas]
    numericas = [col for col in df.columns if col not in categoricas + binarias]
    
    ct = ColumnTransformer([
    ("num", RobustScaler(), numericas),
    ("cat", "passthrough", categoricas),
    ("bin", "passthrough", binarias)
    ])

    df_escalado = ct.fit_transform(df)
    columnas_finales = numericas + categoricas + binarias
    df_escalado = np.array(df_escalado)
    df_escalado = pd.DataFrame(df_escalado, columns=columnas_finales)

    for col in numericas:
        df_escalado[col] = df_escalado[col].astype(float)
        
    catColumnsPos = [df_escalado.columns.get_loc(col) for col in list(df_escalado.select_dtypes("object").columns)]    
    return df_escalado, catColumnsPos
=======
    return data
>>>>>>> 8c4ba22ee0c9efd6662b4f6defff8baf2d8046e8
