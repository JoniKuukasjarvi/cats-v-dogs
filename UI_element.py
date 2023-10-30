import os
import random
import tkinter as tk
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk
import numpy as np


# This sets picture size, and RGB color as the value 3
input_layer = Input(shape=(150, 150, 3))
# Creates convolutional layers (3x3 layer) with multiple different filters
x = Conv2D(16, (3, 3), activation='relu')(input_layer)
# Maxpool takes the highest color values from this previous 3x3 and puts them in a 2x2 layer
x = MaxPooling2D(2, 2)(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
# Flatten reduces the 2x2 layer to one dimensional data, which results to 0 or 1
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=25,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=25,
                                                              class_mode='binary',
                                                              target_size=(150, 150))


root = tk.Tk()
root.title("Cat or Dog Classifier")

img_label = tk.Label(root, text="")
img_label.pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.pack(pady=10)


def classify_image():
    for i in range(10):  # Shows 250 pictures per training process. Could be increased by editing the "batch_size" from earlier
        history = model.fit(train_generator, epochs=1, steps_per_epoch=1,
                            validation_data=validation_generator, validation_steps=1)

    folders = ['D:/Projects/AI_projects/tmp/cats-v-dogs/testing/Dog', 'D:/Projects/AI_projects/tmp/cats-v-dogs/testing/Cat',
               'D:/Projects/AI_projects/tmp/cats-v-dogs/training/Dog', 'D:/Projects/AI_projects/tmp/cats-v-dogs/training/Cat']
    chosen_folder = random.choice(folders)
    files = os.listdir(chosen_folder)
    chosen_images = random.choice(files)
    image_path = chosen_folder + '/' + chosen_images

    img = Image.open(image_path).resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)

    confidence = result[0][0]
    if confidence > 0.5:
        result_var.set(f"It's a Dog! ({int(confidence*100)}% sure)")
    else:
        result_var.set(f"It's a Cat! ({int((1-confidence)*100)}% sure)")

    load = Image.open(image_path)
    render = ImageTk.PhotoImage(load)
    img_label.config(image=render)
    img_label.image = render


def clear_learning():
    global model
    model = None
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy', metrics=['acc'])
    result_var.set("Learning Cleared")


button = tk.Button(root, text="Show Animal", command=classify_image)
button.pack(pady=20)

clear_button = tk.Button(root, text="Clear Learning", command=clear_learning)
clear_button.pack(pady=10)

root.mainloop()
