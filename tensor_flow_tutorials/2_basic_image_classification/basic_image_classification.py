# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Conda ai_env
#     language: python
#     name: ai_env
# ---

# %% slideshow={"slide_type": ""} editable=true deletable=true
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

print(tf.__version__)

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%
train_images.shape

# %%
len(train_labels)

# %%
train_labels

# %%
plt.figure()
plt.imshow(train_images[0])
plt.grid(False)
plt.show()

# %%
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# %%
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# %%
model.fit(
    train_images, 
    train_labels, 
    epochs=10)


# %%
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# %%
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# %%
predictions = probability_model.predict(test_images)

# %%
predictions[0]

# %%
np.argmax(predictions[0])

# %%
test_labels[0]

# %%
def plot_image(i, predictions_aray, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_aray)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                         100*np.max(predictions_aray),
                                         class_names[true_label]),
                                         color=color)

def plot_value_array(i, predictions_aray, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_aray, color='#777777')
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_aray)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

def plot_ith_image(index, predictions_aray, test_labels, test_images, num_rows=1, num_cols=1):
    plt.figure(figsize=(2*2*num_cols,2*num_rows))
    for i in range(num_rows*num_cols):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i+index, predictions_aray[i+index], test_labels, test_images)
        plt.subplot(num_rows,2*num_cols, 2*i+2)
        plot_value_array(i+index, predictions_aray[i+index], test_labels)
    plt.tight_layout()
    plt.show()

# %%
i = 0
plot_ith_image(i, predictions, test_labels, test_images)
predictions[0]

# %%
i = 12
plot_ith_image(i, predictions, test_labels, test_images)
predictions[12]

# %%
plot_ith_image(0, predictions, test_labels, test_images, num_rows=5, num_cols=3)

# %%
img = test_images[1]
print(img.shape)

# %%
img = (np.expand_dims(img, 0))
print(img.shape)

# %%
predictions_single = probability_model.predict(img)
print(predictions_single)

# %%
plot_value_array(1, predictions_single[0], test_labels)
# _ = discarts the value returned
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# %%
np.argmax(predictions_single[0])


# %%
