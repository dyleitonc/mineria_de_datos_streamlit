import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
from PIL import Image
import base64
from io import BytesIO

# Función para cargar el modelo comprimido
def load_model():
    filename = "modelo_entrenado (2).pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo
model = load_model()

# Título y descripción con estilo
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>
        Predicción de Tipo de Cobertura Forestal
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>
        Ingresa las características para predecir el tipo de cobertura forestal.
    </p>
    """,
    unsafe_allow_html=True
)

# Cargar la imagen
image = Image.open("forest.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Mostrar la imagen centrada en HTML
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/jpeg;base64,{img_str}" width="450" />
    </div>
    """,
    unsafe_allow_html=True
)

# Dividir las entradas en columnas (ajustar según las características del dataset)
col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input('Elevation (Elevación)', value=0.0)
    aspect = st.number_input('Aspect (Aspecto de la pendiente)', value=0.0)
    slope = st.number_input('Slope (Pendiente)', value=0.0)
    horizontal_distance_to_hydrology = st.number_input('Distancia Horizontal a Hidrología', value=0.0)
    vertical_distance_to_hydrology = st.number_input('Distancia Vertical a Hidrología', value=0.0)
    horizontal_distance_to_roadways = st.number_input('Distancia Horizontal a Carreteras', value=0.0)

with col2:
    hillshade_9am = st.number_input('Hillshade 9am', value=0.0)
    hillshade_noon = st.number_input('Hillshade Noon', value=0.0)
    hillshade_3pm = st.number_input('Hillshade 3pm', value=0.0)
    horizontal_distance_to_fires = st.number_input('Distancia Horizontal a Fuegos', value=0.0)
    wilderness_area = st.selectbox('Área Wilderness', ['Wilderness Area 1', 'Wilderness Area 2', 'Wilderness Area 3', 'Wilderness Area 4'])
    soil_type = st.selectbox('Tipo de Suelo', ['Soil Type 1', 'Soil Type 2', 'Soil Type 3', 'Soil Type 4'])

# Convertir las variables categóricas a índices
wilderness_area = ['Wilderness Area 1', 'Wilderness Area 2', 'Wilderness Area 3', 'Wilderness Area 4'].index(wilderness_area)
soil_type = ['Soil Type 1', 'Soil Type 2', 'Soil Type 3', 'Soil Type 4'].index(soil_type)

# Crear un DataFrame con las entradas
datos_usuario = pd.DataFrame({
    'Elevation': [elevation],
    'Aspect': [aspect],
    'Slope': [slope],
    'Horizontal_Distance_To_Hydrology': [horizontal_distance_to_hydrology],
    'Vertical_Distance_To_Hydrology': [vertical_distance_to_hydrology],
    'Horizontal_Distance_To_Roadways': [horizontal_distance_to_roadways],
    'Hillshade_9am': [hillshade_9am],
    'Hillshade_Noon': [hillshade_noon],
    'Hillshade_3pm': [hillshade_3pm],
    'Horizontal_Distance_To_Fires': [horizontal_distance_to_fires],
    'Wilderness_Area': [wilderness_area],
    'Soil_Type': [soil_type]
})

# Botón para predecir
if st.button('🚀 Predecir Tipo de Cobertura Forestal'):
    # Realizar la predicción
    prediccion = model.predict(datos_usuario.values)
    
    # Mostrar la predicción con estilo
    st.markdown(
        f"""
        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white;'>El tipo de cobertura forestal predicho es:</h2>
            <h1 style='color: white;'>Clase {prediccion[0]}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Barra lateral con información adicional
with st.sidebar:
    st.markdown("### Información Adicional")
    st.markdown(""" 
        Este modelo predice el tipo de cobertura forestal basado en características geográficas y ambientales.
    """)

    # Mostrar los hiperparámetros del modelo (si aplica)
    try:
        hiperparametros = model.get_params()
        descripcion_hiperparametros = f"""
        ### Hiperparámetros del Modelo:
        **Regresor (Random Forest):**
        - **Número de árboles**: {hiperparametros['n_estimators']}  
        - **Profundidad máxima**: {hiperparametros['max_depth']}  
        """
        st.markdown(descripcion_hiperparametros)
    except AttributeError:
        st.markdown("El modelo no tiene hiperparámetros accesibles.")
