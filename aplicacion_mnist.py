import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gzip
import pickle

# Crear un directorio para guardar las imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def preprocess_image(image):
    """Preprocesa la imagen para que sea compatible con el modelo."""
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = img_to_array(image) / 255.0  # Normalizar los píxeles
    image_array = image_array.reshape(1, -1)  # Convertir a vector de 784 características
    return image_array

def main():
    # Estilos personalizados
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .description {
            font-size: 18px;
            color: #555555;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            font-size: 14px;
            color: #888888;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Título y descripción
    st.markdown('<div class="main-title">Clasificación de Dígitos MNIST</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Sube una imagen de un dígito y la clasificaremos usando un modelo preentrenado.</div>', unsafe_allow_html=True)

    # Widget de subida de archivos
    uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        st.subheader("Vista previa de la imagen subida")
        image = Image.open(uploaded_file)

        # Procesar la imagen
        preprocessed_image = preprocess_image(image)

        # Mostrar imágenes antes y después del preprocesamiento
        st.subheader("Imágenes antes y después del preprocesamiento")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen original", use_container_width=True, output_format="auto")
        with col2:
            st.image(preprocessed_image.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True, output_format="auto")

        # Guardar la imagen
        file_path = save_image(uploaded_file)
        st.success(f"Imagen guardada")

        # Diccionario de clases para MNIST
        mnist_classes = {i: str(i) for i in range(10)}

        # Botón para clasificar la imagen
        if st.button("Clasificar imagen"):
            with st.spinner("Cargando modelo y clasificando..."):
                model = load_model()
                prediction = model.predict(preprocessed_image)
                
                # Verificar valores de predicción
                st.success(f"La imagen fue clasificada como: {prediction}")

    # Footer
    st.markdown('<div class="footer">© 2025 - Clasificación de imágenes con Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

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
