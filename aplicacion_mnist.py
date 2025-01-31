import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Cargar y preprocesar los datos MNIST
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalizar las imágenes
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    # Convertir las etiquetas a formato one-hot
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return (train_images, train_labels), (test_images, test_labels)

# Cargar los datos
(train_images, train_labels), (test_images, test_labels) = load_data()

# Construir el modelo
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Aplanar las imágenes de 28x28 a un vector de 784 elementos
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 clases de dígitos (0-9)
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Inicializar el modelo
model = create_model()

# Entrenar el modelo
def train_model():
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Botón para entrenar el modelo
if st.button('Entrenar Modelo'):
    train_model()
    st.success("¡Modelo entrenado exitosamente!")

# Función para predecir y mostrar el resultado
def predict_digit(image):
    image = np.expand_dims(image, axis=0)  # Añadir la dimensión del batch
    image = image / 255.0  # Normalizar la imagen
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Mostrar una imagen aleatoria de prueba
def display_random_image():
    index = np.random.randint(0, test_images.shape[0])
    image = test_images[index]
    label = np.argmax(test_labels[index])
    
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Dígito Real: {label}')
    ax.axis('off')
    st.pyplot(fig)
    
    return image, label

# Botón para predecir un dígito
if st.button('Predecir un Dígito'):
    image, true_label = display_random_image()
    predicted_label = predict_digit(image)
    
    # Mostrar la predicción
    st.markdown(f'Predicción del modelo: {predicted_label}')
    st.markdown(f'Dígito real: {true_label}')
    
    # Calcular la precisión
    if predicted_label == true_label:
        st.success("¡Predicción Correcta!")
    else:
        st.error("Predicción Incorrecta")

# Información sobre el modelo y los datos
st.sidebar.markdown("### Modelo de Red Neuronal para Predicción de Dígitos MNIST")
st.sidebar.markdown("""
    El modelo que utiliza esta aplicación es una red neuronal simple con una capa oculta de 128 neuronas. 
    El conjunto de datos MNIST contiene imágenes de dígitos escritos a mano, que son usadas para entrenar el modelo. 
    El modelo realiza la predicción de los dígitos a partir de las imágenes.
""")
