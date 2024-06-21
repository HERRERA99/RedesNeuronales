import tensorflow as tf
import numpy as np
import os

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenhein = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Defino el modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilo el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entreno el modelo
print("Comenzamos a entrenar ...")
historial = modelo.fit(celsius, fahrenhein, epochs=1000, verbose=False)
print("Modelo entrenado!")

# Guardar el modelo
modelo.save('C:\\Users\\aitor\\PycharmProjects\\RedesNeuronales\\modelo_conversion_temp.h5')

# Confirmar que el archivo ha sido guardado
if os.path.exists('C:\\Users\\aitor\\PycharmProjects\\RedesNeuronales\\modelo_conversion_temp.h5'):
    print("El modelo ha sido guardado correctamente.")
else:
    print("Error al guardar el modelo.")

# Miro la grafica de errores
import matplotlib.pyplot as plt
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de pérdidas')
plt.plot(historial.history['loss'])
plt.show()

