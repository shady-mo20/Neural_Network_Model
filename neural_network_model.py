import tensorflow as tf
from tensorflow.keras import layers, models


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') ])


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=5, batch_size=64)

train_acc = history.history['accuracy'][-1]
print(f"Train acc: {train_acc}")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test acc: {test_acc}")
