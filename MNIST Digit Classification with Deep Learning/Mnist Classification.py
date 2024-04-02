import tensorflow as tf 
from tensorflow import keras 
from keras import layers 
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#  Load the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the shapes of the vectors
print('X_train shape:', x_train.shape)
print('Y_train shape:', y_train.shape)
print('X_test shape:', x_test.shape)
print('Y_test shape:', y_test.shape)

# Data visualisation Plot a few sample images 
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')  # Hide axes
plt.show() 


# Rescaling the data 

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# Build the model 
inputs = layers.Input(shape=(28,28))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation= "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation= "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation= "relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs= layers.Dense(10, activation= "softmax")(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(loss = keras.losses.SparseCategoricalCrossentropy(),
              optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.98),
              metrics= ['accuracy'],)


# Train the model 
history = model.fit(x_train, y_train, validation_split=0.2, batch_size= 32, epochs= 100, verbose=2)

print('=============Evaluation ===============')
# Evaluate the model 
predict = model.evaluate(x_test,y_test, batch_size=32, verbose=2)

#==============================visualisation========================

# Visualize accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

# Visualize loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


print('============= Prediction ===============')
# Prediction
num_samples_to_visualize = 10
subset_indices = np.random.choice(len(x_test), num_samples_to_visualize, replace=False)
predictions = model.predict(x_test[subset_indices])
predicted_labels = np.argmax(predictions, axis=1)

# Visualize subset of test samples with predicted labels
plt.figure(figsize=(12, 6))
for i, idx in enumerate(subset_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f'Predicted: {predicted_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

