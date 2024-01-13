import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.load_model('handwritten.model')
loss, accuracy = model.evaluate(x_test, y_test)

predictions = model.predict(x_train)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_train

# Calculate precision
precision = precision_score(true_labels, predicted_labels, average='weighted')

print(f'Precision: {precision}')
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
