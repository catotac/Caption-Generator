#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rahulsn
"""

import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import model_from_json
from keras.utils import to_categorical
from scipy import misc

import os
import model as ml

def images_pixels(path_data, data, shape):
    pixels = {}
    imagefiles = sorted(os.listdir(path_data))
    for files in imagefiles:
        if files.split('.jpg')[0] in data:
            image_file = path_data + files
            image = load_img(image_file, target_size=(shape, shape))
            image = img_to_array(image)
            pixels[files.split('.jpg')[0]] = image
    return pixels

def get_annotations(file, images):
    f = open(file, 'r')
    data = f.readlines()
    descriptions = {}
    for i in range(len(data)):
        image_desc = data[i].split('\t')
        imageId = re.split('.jpg',image_desc[0])[0]
        desr = image_desc[1].rstrip()
        while desr[-1] == " " or desr[-1] == '.':
            desr = desr[:-1]
        if imageId in images:
            if imageId not in descriptions:
                descriptions[imageId] = []
                descriptions[imageId].append(desr)
            else:
                descriptions[imageId].append(desr)
    return descriptions
def get_trainingimages(file):
    f = open(file, 'r')
    Images = f.read().split('\n')
    train = [d.split('.jpg')[0] for d in Images]
    return train
def image_files(file, **kwargs):
    image_files = sorted(os.listdir(file))
    files = []
    for i in range(len(image_files)):
        imageId = re.split('.jpg',image_files[i])[0]
        files.append(imageId)
    return files
def modify_annotation(Image_description):
    imageIds = list(Image_description.keys())
    split_description = {}
    temp = []
    for i in range(len(imageIds)):
        split_description[imageIds[i]] = []
        for j in range(len(Image_description[imageIds[i]])):
            temp = Image_description[imageIds[i]][j].split()
            desc = 'startseq ' + ' '.join(temp) + ' endseq'
            split_description[imageIds[i]].append(desc)
    return split_description
def text_tokenizer(image_description):
    t = Tokenizer()
    all_annotations = []
    imageIds = list(image_description.keys())
    for i in range(len(imageIds)):
        for d in image_description[imageIds[i]]:
           all_annotations.append(d)
    t.fit_on_texts(all_annotations)
    return t
def max_length(descriptions):
    all_annotations = []
    imageIds = list(image_description.keys())
    for i in range(len(imageIds)):
        for d in image_description[imageIds[i]]:
           all_annotations.append(d)
    return max(len(d.split()) for d in all_annotations)
def list2array(array, shape):
    sampleSize = len(array)
    print(sampleSize)
    returnvalue = np.empty([sampleSize, shape, shape, 3], dtype = np.float32)
    for i in range(sampleSize):
        returnvalue[i] = array[i]
    return returnvalue

# Preparing datasets for small input size
# =============================================================================
# def data_preparation(t, max_len, descriptions, pixels):
#     X_images, temp_img = [], []
#     X_sequence, temp_seq = [], []
#     Y_output, temp_out = [], []
#     vocab_size = len(t.word_index) + 1
#     imageIds = list(descriptions.keys())
#     for i in range(len(imageIds)):
#         for j in range(len(descriptions[imageIds[i]])):
#             desc = t.texts_to_sequences(descriptions[imageIds[i]])[0]
#             for i in range(1, len(desc)):
#                 seq1 = pad_sequences([desc[:i]], maxlen=max_len)[0]
#                 seq2 = to_categorical([desc[i]], num_classes=vocab_size)[0]
#                 temp_img.append(pixels[imageIds[i]])
#                 temp_seq.append(seq1)
#                 temp_out.append(seq2)
#         X_images.append(np.array(temp_img))
#         X_sequence.append(np.array(temp_seq))
#         Y_output.append(np.array(temp_out))
#         temp_img, temp_seq, temp_out = [], [], []
# #    X_images = list2array(X_images)
#     return X_images,X_sequence, Y_output
# =============================================================================

def input_preparation(t, max_len, descriptions, pixels, shape):
    X_images = []
    X_sequence = []
    Y_output = []
    vocab_size = len(t.word_index) + 1
    imageIds = list(descriptions.keys())
    for i in range(len(imageIds)):
        for j in range(len(descriptions[imageIds[i]])):
            desc = t.texts_to_sequences(descriptions[imageIds[i]])[0]
            for i in range(1, len(desc)):
                seq1 = pad_sequences([desc[:i]], maxlen=max_len)[0]
                seq2 = to_categorical([desc[i]], num_classes=vocab_size)[0]
                X_images.append(pixels[imageIds[i]])
                X_sequence.append(seq1)
                Y_output.append(seq2)
    X_images = list2array(X_images, shape)
    return X_images,np.array(X_sequence), np.array(Y_output)

def data_generator(photos, descriptions, outputs):
    while 1:
        for i in range(10, len(photos), 10):
            if i + 10 > len(photos):
                i = len(photos)
            x1 = np.array(photos[i-10:i])
            x2 = np.array(descriptions[i-10:i])
            y1 = np.array(outputs[i-10:i])
            yield [[x1, x2], y1]
                

shape = 64
path =  os.path.dirname(os.path.realpath(__file__))
total_images = image_files(path + '/data/Flicker8k_Dataset/')
training_file = path + '/data/Flickr8k_text/Flickr_8k.trainImages.txt'
test_file = path + '/data/Flickr8k_text/Flickr_8k.devImages.txt'

# Import training and test data
training_images = get_trainingimages(training_file)
test_images = get_trainingimages(test_file)

# Import all the annotations
annotation_file =  path + '/data/Flickr8k_text/Flickr8k.lemma.token.txt'
image_description = get_annotations(annotation_file, training_images)

# Dictionary containing the captions of each image. dict[image] = "annotations"
test_imagedescription = get_annotations(annotation_file, test_images)
annotations = modify_annotation(image_description)

# Add "<START>" and "<END>" tag
test_annotations = modify_annotation(test_imagedescription)
tokenizer = text_tokenizer(annotations)

vocab_size = len(tokenizer.word_index) + 1 # Vocabulary size
max_len = max_length(annotations) # Maximum length of annotations
# Data preparation
training_pixels = images_pixels(path + '/data/Flicker8k_Dataset/', list(image_description.keys()), shape)
test_pixels = images_pixels(path + '/data/Flicker8k_Dataset/', list(test_imagedescription.keys()), shape)
X1, X2, y = input_preparation(tokenizer, max_len, annotations, training_pixels, shape)
X1test, X2test, ytest = input_preparation(tokenizer, max_len, test_annotations, test_pixels, shape)

# Creating LSTM + CNN model 
model = ml.lstm_model(vocab_size, max_len, shape)
# Training Starts
model.fit([X1, X2, y, epochs=1, verbose=2, validation_data=([X1test, X2test, ytest))

# Save model to j_son
model_json = model.to_json()
with open("model_cnn.json", "w") as json_file:
    json_file.write(model_json)

# Save weights
model.save_weights("model_cnn.h5")


# Data Fitting using data generator (Useful for CPU with not enough memory)

# =============================================================================
# epochs = 20
# steps = len(annotations)
# for i in range(epochs):
# 	# create the data generator
# 	generator = data_generator(X1, X2, y)
# 	# fit for one epoch
# 	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# 	# save model
# 	model.save('model_' + str(i) + '.h5') 
# =============================================================================
