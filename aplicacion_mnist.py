import streamlit as st
from PIL import Image
import os
import numpy as np
import gzip
import pickle
from tensorflow.keras.preprocessing.image import img_to_array

# Crear un directorio para guardar las imágenes
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image(uploaded_file):
    """Guarda la imagen subida en el directorio UPLOAD_FOLDER."""
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_model():
    """Cargar el modelo desde un archivo comprimido."""
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
            font-size: 36px;
            font-weight: bold;
            color: #E63946;
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

    # Mostrar una imagen de encabezado
    #st.image("mnist_banner.jpg", use_container_width=True)

    # Título y descripción
    st.markdown('<div class="main-title">Clasificación de Dígitos MNIST</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Sube una imagen de un dígito y la clasificaremos con un modelo de aprendizaje automático.</div>', unsafe_allow_html=True)

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
            st.image(image, caption="Imagen original", use_container_width=True)
        with col2:
            st.image(preprocessed_image.reshape(28, 28), caption="Imagen preprocesada", use_container_width=True)

        # Guardar la imagen
        save_image(uploaded_file)
        
        # Botón para clasificar la imagen
        if st.button("🔍 Clasificar imagen"):
            with st.spinner("Cargando modelo y clasificando..."):
                model = load_model()
                prediction = model.predict(preprocessed_image)
                st.success(f"La imagen fue clasificada como: {prediction}")
    
    # Mostrar precisión del modelo
    model_accuracy = 0.8717
    st.markdown(f'<p style="font-size: 24px;">📊 Exactitud del modelo: <strong>{model_accuracy * 100:.2f}%</strong></p>', unsafe_allow_html=True)

    # Columna para los hiperparámetros
    col1, col2 = st.columns([3, 1])  # Hacer la columna de los hiperparámetros más estrecha

    with col2:
        # Hiperparámetros del modelo
        hiperparametros = {
            'n_neighbors': 4,
            'p': 3,
            'metric': 'minkowski',
            'weights': 'uniform'
        }
        
        # Explicación de los hiperparámetros
        st.markdown("### ⚙️ Hiperparámetros del Modelo")
        st.markdown("Los hiperparámetros son configuraciones que afectan el rendimiento del modelo. Aquí explicamos algunos de ellos:")
        st.markdown(f"- **Número de vecinos (`n_neighbors`)**: {hiperparametros['n_neighbors']} - Controla cuántos vecinos se usan para clasificar un nuevo dígito.")
        st.markdown(f"- **Parámetro de distancia (`p`)**: {hiperparametros['p']} - Define cómo se mide la distancia entre imágenes. Un valor de 3 usa la distancia de Minkowski.")
        st.markdown(f"- **Métrica (`metric`)**: {hiperparametros['metric']} - Define la fórmula matemática usada para calcular la similitud entre imágenes.")
        st.markdown(f"- **Pesos (`weights`)**: {hiperparametros['weights']} - Indica si todos los vecinos tienen la misma importancia en la clasificación.")

    # Footer
    st.markdown('<div class="footer">© 2025 - Clasificación de imágenes con Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
