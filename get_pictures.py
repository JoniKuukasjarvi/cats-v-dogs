import os
import zipfile
import random
import tensorflow as tf
from shutil import copyfile


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "D:/Projects/AI_projects/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/training/Cat/"
TESTING_CATS_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/testing/Cat/"
DOG_SOURCE_DIR = "D:/Projects/AI_projects/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/training/Dog/"
TESTING_DOGS_DIR = "D:/Projects/AI_projects/tmp/cats-v-dogs/testing/Dog/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


# Seems like there are more dogs than cats in these files
