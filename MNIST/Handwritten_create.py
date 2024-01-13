import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets

model = tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"I think the number is {np.argmax(prediction)}")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()
    image_number += 1
