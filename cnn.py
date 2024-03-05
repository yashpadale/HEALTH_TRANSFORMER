
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


import os
import cv2
import numpy as np

medical_images_dir = 'data/medical_images'
natural_images_dir = 'data/natural_images'
scanned_documents_dir = 'data/scanned_documents'

def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        # Load the image
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        images.append(image)
        labels.append(label)
    return images, labels

medical_images, medical_labels = load_and_preprocess_images(medical_images_dir, label=0)
natural_images, natural_labels = load_and_preprocess_images(natural_images_dir, label=1)
scanned_documents, scanned_labels = load_and_preprocess_images(scanned_documents_dir, label=2)
classes=['medical_images','natural_images','scanned_documents']
all_images = medical_images + natural_images + scanned_documents
all_labels = medical_labels + natural_labels + scanned_labels

X = np.array(all_images)
y = np.array(all_labels)


indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_split = int(0.8 * len(X))
val_split = int(0.1 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:train_split + val_split], y[train_split:train_split + val_split]
X_test, y_test = X[train_split + val_split:], y[train_split + val_split:]


print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)



input_shape = (224, 224, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


def load_and_preprocess_image(file_location):
    image = cv2.imread(file_location)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image

def predict_image(file_location, model):
    image = load_and_preprocess_image(file_location)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions



image_file_location = 'img985.jpg'


predictions = predict_image(image_file_location, model)

print("Predicted Probabilities:", predictions)
m=max(predictions[0])
f=predictions[0]
for i in range(len(f)):
    if f[i]==m:
        print(f'Detected class is {classes[i]}')