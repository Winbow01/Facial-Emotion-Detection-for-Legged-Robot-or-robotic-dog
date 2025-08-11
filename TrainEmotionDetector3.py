
# import required packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import class_weight
from keras.callbacks import ReduceLROnPlateau
from keras.applications import NASNetLarge
from keras.applications import InceptionResNetV2


# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)


# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(75, 75),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(75, 75),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')


# Get the class labels
train_labels = train_generator.classes
validation_labels = validation_generator.classes
# Convert categorical labels to one-hot encoded vectors
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels),
                                                  y=train_labels)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)



base_model  = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_model.add(Dropout(0.5))
#
# emotion_model.add(Flatten())
# emotion_model.add(Dense(7, activation='softmax'))

# cv2.ocl.setUseOpenCL(False)


model = Model(inputs=base_model .input, outputs=predictions)



# emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="checkpoint/",
                                                        save_weights_only=False,
                                                        save_best_only=True,
                                                        save_freq="epoch",
                                                        verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                patience=8, min_lr=0.00001)

# emotion_model.summary()
# Train the neural network/model
emotion_model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64,
        callbacks=[reduce_lr])

# save model structure in jason file
model_json = model.to_json()
with open("model/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save_weights('emotion_model.h5')




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

#GlobalAveragePooling2D()
# 448/448 [==============================] - 40s 89ms/step - loss: 0.0134 - accuracy: 0.9954 - val_loss: 3.6122 - val_accuracy: 0.6482 - lr: 2.0000e-04
# Epoch 50/50
# 448/448 [==============================] - 40s 89ms/step - loss: 0.0093 - accuracy: 0.9970 - val_loss: 3.5445 - val_accuracy: 0.6512 - lr: 4.0000e-05
