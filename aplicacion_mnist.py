# Barra lateral para los hiperparámetros
with st.sidebar:
    st.markdown("### ⚙️ Hiperparámetros del Modelo")
    hiperparametros = {
        'n_neighbors': 4,
        'p': 3,
        'metric': 'minkowski',
        'weights': 'uniform'
    }

    # Explicación de los hiperparámetros
    st.markdown("Estos son los hiperparámetros clave que afectan el rendimiento del modelo de clasificación:")
    st.markdown(f"- **Número de vecinos (`n_neighbors`)**: {hiperparametros['n_neighbors']} - Este parámetro controla cuántos vecinos se consideran al clasificar un nuevo dígito.")
    st.markdown(f"- **Parámetro de distancia (`p`)**: {hiperparametros['p']} - Define cómo se mide la distancia entre las imágenes. Un valor de 3 indica el uso de la distancia de Minkowski.")
    st.markdown(f"- **Métrica de distancia (`metric`)**: {hiperparametros['metric']} - Especifica la fórmula utilizada para medir la similitud entre las imágenes. En este caso, se utiliza la métrica de Minkowski.")
    st.markdown(f"- **Pesos (`weights`)**: {hiperparametros['weights']} - Este parámetro determina si todos los vecinos tienen la misma importancia en la clasificación.")
