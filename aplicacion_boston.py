import streamlit as st
import pandas as pd
import numpy as np
import gzip
import pickle
from PIL import Image

# Función para cargar el modelo comprimido
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar el modelo
model = load_model()

# Título y descripción con estilo
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>
        Predicción del Valor de una Casa
    </h1>
    <p style='text-align: center; font-size: 18px; color: #666;'>
        Ingresa las características de la casa para obtener una estimación de su valor:
    </p>
    """,
    unsafe_allow_html=True
)

# Cargar la imagen
image = Image.open('casa.jpg')

# Mostrar la imagen con un tamaño de ancho específico (por ejemplo, 500 píxeles)
st.image(image, use_container_width=True, width=250)  # Ajusta el valor de width según sea necesario

# Dividir las entradas en columnas
col1, col2 = st.columns(2)

with col1:
    crim = st.number_input('CRIM (Tasa de criminalidad)', value=0.0)
    zn = st.number_input('ZN (Proporción de terreno residencial)', value=0.0)
    indus = st.number_input('INDUS (Proporción de acres de negocios)', value=0.0)
    chas = st.number_input('CHAS (Limita con el río Charles: 1 si, 0 no)', value=0)
    nox = st.number_input('NOX (Concentración de óxidos de nitrógeno)', value=0.0)
    rm = st.number_input('RM (Número promedio de habitaciones)', value=0.0)

with col2:
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
if st.button('🚀 Predecir Valor de la Casa'):
    # Realizar la predicción
    prediccion = model.predict(datos_usuario.values)
    
    # Mostrar la predicción con estilo
    st.markdown(
        f"""
        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white;'>El valor predicho de la casa es:</h2>
            <h1 style='color: white;'>${prediccion[0]:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Barra lateral con información adicional
with st.sidebar:
    st.markdown("### Información Adicional")
    st.markdown("""
        - **CRIM**: Tasa de criminalidad per cápita.
        - **ZN**: Proporción de terreno residencial.
        - **INDUS**: Proporción de acres de negocios.
        - **CHAS**: Limita con el río Charles (1 si, 0 no).
        - **NOX**: Concentración de óxidos de nitrógeno.
        - **RM**: Número promedio de habitaciones.
        - **AGE**: Proporción de unidades construidas antes de 1940.
        - **DIS**: Distancia a centros de empleo.
        - **RAD**: Índice de accesibilidad a autopistas.
        - **TAX**: Tasa de impuesto sobre la propiedad.
        - **PTRATIO**: Relación alumno-maestro.
        - **B**: Proporción de personas de ascendencia afroamericana.
        - **LSTAT**: Porcentaje de población de estatus bajo.
    """)


    # Explicación de los hiperparámetros y sus valores
    hiperparametros = model.get_params()

    descripcion_hiperparametros = f"""
    ### Hiperparámetros del Modelo:
    
    **Escalador (StandardScaler):**
    - **`scaler__copy`**: {hiperparametros['steps'][0][1].get_params()['copy']}  
      Esto asegura que no se modifique el conjunto de datos original durante el escalado.
    - **`scaler__with_mean`**: {hiperparametros['steps'][0][1].get_params()['with_mean']}  
      Indica si se debe centrar la variable al restarle la media (es útil para normalizar).
    - **`scaler__with_std`**: {hiperparametros['steps'][0][1].get_params()['with_std']}  
      Especifica si se debe dividir por la desviación estándar, lo que permite que los datos estén escalados.

    **Regresor (KernelRidge):**
    - **`reg__alpha`**: {hiperparametros['steps'][1][1].get_params()['alpha']}  
      Es el parámetro de regularización que controla la complejidad del modelo: valores más altos previenen sobreajuste.
    - **`reg__coef0`**: {hiperparametros['steps'][1][1].get_params()['coef0']}  
      Este parámetro ajusta la influencia del término de sesgo en el modelo.
    - **`reg__degree`**: {hiperparametros['steps'][1][1].get_params()['degree']}  
      Define el grado del polinomio para el kernel, afectando la flexibilidad del modelo.
    - **`reg__kernel`**: {hiperparametros['steps'][1][1].get_params()['kernel']}  
      El kernel 'rbf' es utilizado para medir la similitud entre los puntos de datos en el espacio de características.
    """

    # Mostrar la descripción con los valores de los hiperparámetros
    with st.expander("Ver hiperparámetros del modelo"):
        st.markdown(descripcion_hiperparametros)
