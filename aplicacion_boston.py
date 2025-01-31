import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle

# Título de la aplicación
st.title('Predicción del valor de una casa')

# Función para cargar el modelo comprimido
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo
model = load_model()

# Entradas del usuario
st.header('Ingresa las características de la casa')
crim = st.number_input('CRIM (Tasa de criminalidad)', value=0.0)
zn = st.number_input('ZN (Proporción de terreno residencial)', value=0.0)
indus = st.number_input('INDUS (Proporción de acres de negocios)', value=0.0)
chas = st.number_input('CHAS (Limita con el río Charles: 1 si, 0 no)', value=0)
nox = st.number_input('NOX (Concentración de óxidos de nitrógeno)', value=0.0)
rm = st.number_input('RM (Número promedio de habitaciones)', value=0.0)
age = st.number_input('AGE (Proporción de unidades construidas antes de 1940)', value=0.0)
dis = st.number_input('DIS (Distancia a centros de empleo)', value=0.0)
rad = st.number_input('RAD (Índice de accesibilidad a autopistas)', value=0.0)
tax = st.number_input('TAX (Tasa de impuesto sobre la propiedad)', value=0.0)
ptratio = st.number_input('PTRATIO (Relación alumno-maestro)', value=0.0)
b = st.number_input('B (Proporción de personas de ascendencia afroamericana)', value=0.0)
lstat = st.number_input('LSTAT (Porcentaje de población de estatus bajo)', value=0.0)

# Crear un DataFrame con las entradas
datos_usuario = pd.DataFrame({
    'CRIM': [crim],
    'ZN': [zn],
    'INDUS': [indus],
    'CHAS': [chas],
    'NOX': [nox],
    'RM': [rm],
    'AGE': [age],
    'DIS': [dis],
    'RAD': [rad],
    'TAX': [tax],
    'PTRATIO': [ptratio],
    'B': [b],
    'LSTAT': [lstat]
})

# Botón para predecir
if st.button('Predecir valor de la casa'):
    # Realizar la predicción
    prediccion = model.predict(datos_usuario.values)
    
    # Mostrar la predicción
    st.success(f'El valor predicho de la casa es: **${prediccion[0]:.2f}**')
