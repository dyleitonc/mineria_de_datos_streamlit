import streamlit as st
from PIL import Image
import os
import numpy as np
import gzip
import pickle

# Crear un directorio para guardar las imágenes si no existe
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Función para cargar el modelo
@st.cache_resource
def load_model():
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Cargar modelo
model = load_model()

# Hiperparámetros del modelo
accuracy = 0.8717  # Precisión del modelo
hyperparameters = {
    'n_neighbors': 4,
    'p': 3,
    'algorithm': 'auto',
    'leaf_size': 30,
    'metric': 'minkowski',
    'weights': 'uniform'
}

def save_image(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def preprocess_image(image):
    image = image.convert('L')  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = image_array.reshape(1, -1)  # Convertir a vector
    return image_array

# Diseño mejorado con CSS
st.markdown(
    """
    <style>
        .title { text-align: center; font-size: 32px; color: #2E86C1; }
        .description { text-align: center; font-size: 18px; color: #555; }
        .info-box { background-color: #E8F6F3; padding: 15px; border-radius: 10px; }
        .footer { text-align: center; font-size: 14px; color: #888; margin-top: 50px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown('<div class="title">Clasificación de Dígitos MNIST</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Sube una imagen de un dígito y la clasificaremos usando un modelo de KNN.</div>', unsafe_allow_html=True)

# Mostrar información del modelo en una caja
st.markdown(f"""
<div class='info-box'>
    <b>Precisión del modelo:</b> {accuracy * 100:.2f}%<br>
    <b>Hiperparámetros:</b>
    <ul>
        <li><b>Vecinos:</b> {hyperparameters['n_neighbors']}</li>
        <li><b>P:</b> {hyperparameters['p']}</li>
        <li><b>Algoritmo:</b> {hyperparameters['algorithm']}</li>
        <li><b>Métrica:</b> {hyperparameters['metric']}</li>
        <li><b>Leaf Size:</b> {hyperparameters['leaf_size']}</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Subida de imágenes
uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)

    # Mostrar imágenes en columnas
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen original", use_column_width=True)
    with col2:
        st.image(preprocessed_image.reshape(28, 28), caption="Imagen preprocesada", use_column_width=True)

    # Botón para predecir
    if st.button("Clasificar imagen"):
        prediction = model.predict(preprocessed_image)
        st.success(f"La imagen fue clasificada como: {prediction[0]}")

# Footer
st.markdown('<div class="footer">© 2025 - Clasificación de imágenes con Streamlit</div>', unsafe_allow_html=True)


