import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenhein = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss='mean_squared_error'
)

print("Comenzamos a entrenar ...")
historial = modelo.fit(celsius, fahrenhein, epochs=1000, verbose=False)
print("Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de pérdidas')
plt.plot(historial.history['loss'])
