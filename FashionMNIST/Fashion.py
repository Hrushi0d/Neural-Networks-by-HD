import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images = test_images / 255.0
model = tf.keras.models.load_model('conv-fashionable-2.model')

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


for i in range(5):
    ans = np.argmax(predictions[i])
    print(f"I think its {class_names[ans]}")
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.show()