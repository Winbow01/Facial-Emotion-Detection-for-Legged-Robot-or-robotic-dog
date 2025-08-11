
# import required packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from sklearn.utils import class_weight

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)


# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


# Get the class labels
train_labels = train_generator.classes
validation_labels = validation_generator.classes
# Convert categorical labels to one-hot encoded vectors
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels),
                                                  y=train_labels)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(keras.layers.BatchNormalization())

emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(keras.layers.BatchNormalization())
# reg_wdecay = keras.regularizers.l2(0.05)

emotion_model.add(Flatten())
emotion_model.add(Dense(2048, activation='relu',  kernel_regularizer=None))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
# emotion_model.summary()
# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=90,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("model/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')




###################     test   #################################################
import matplotlib.pyplot as plt

# Create a matplotlib figure  
fh = plt.figure(figsize=(8,4))

# Add second plot to the 1x2 figure
ph = fh.add_subplot(1,1,1)
# ...the values of loss over epochs
ph.plot(emotion_model_info.history['loss'], label='loss', color='tab:orange')
ph.plot(emotion_model_info.history['val_loss'], label='val_loss', color='tab:green')
ph.plot(emotion_model_info.history['accuracy'], label='accuracy', color='tab:red')
ph.plot(emotion_model_info.history['val_accuracy'], label='val_accuracy', color='tab:blue')
ph.set_xlabel('Epoch')
ph.set_ylabel('accuracy, val_accuracy, loss and val_accuracy')
ph.set_title('Training history')

# Disable interactive plot mode
plt.ioff()
# Now, calling plot show how plot block the execution of the script until the figure is closed
plt.show()

#  120   ; 64-128-...
# Epoch 119/120
# 448/448 [==============================] - 10s 21ms/step - loss: 0.0696 - accuracy: 0.9744 - val_loss: 1.8136 - val_accuracy: 0.6438
# Epoch 120/120
# 448/448 [==============================] - 10s 21ms/step - loss: 0.0678 - accuracy: 0.9750 - val_loss: 1.8675 - val_accuracy: 0.6341

#90  64 。。。- 2048
# Epoch 89/90
# 448/448 [==============================] - 26s 58ms/step - loss: 0.0720 - accuracy: 0.9746 - val_loss: 1.6170 - val_accuracy: 0.6422
# Epoch 90/90
# 448/448 [==============================] - 26s 58ms/step - loss: 0.0613 - accuracy: 0.9779 - val_loss: 1.6202 - val_accuracy: 0.6353