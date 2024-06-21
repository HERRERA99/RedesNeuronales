import tensorflow as tf
import numpy as np
import os

model_path = 'C:\\Users\\aitor\\PycharmProjects\\RedesNeuronales\\modelo_conversion_temp.h5'

# Confirmar que el archivo del modelo existe
if not os.path.exists(model_path):
    print("El archivo del modelo no se encontró. Asegúrate de que el archivo esté en el directorio correcto.")
    exit()

# Cargar el modelo guardado
modelo = tf.keras.models.load_model(model_path)

# Realizar una predicción
celsius_input = float(input("Introduce la temperatura en grados Celsius: "))
result = modelo.predict(np.array([celsius_input]))
print("El resultado es " + str(result) + " fahrenheit!")
