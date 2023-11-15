import re
import cv2
import numpy as np
import os
from keras import backend as K
from crop_data_utils import parse_image_filename

#funtion to examine the dataset and check for issues in files
def count_files_in_folders(*folder_paths):
    total_files = 0
    # Ensure an even number of folder paths
    if len(folder_paths) % 2 != 0:
        raise ValueError("Folder paths must be provided in pairs.")
    for i in range(0, len(folder_paths), 2):
        folder_0 = folder_paths[i]
        folder_1 = folder_paths[i + 1]
        _, _, files_0 = next(os.walk(folder_0))
        _, _, files_1 = next(os.walk(folder_1))
        file_count_0 = len(files_0)
        file_count_1 = len(files_1)
        file_count = file_count_0 + file_count_1
        total_files += file_count
        print(f'Total files in Folder Pair {i // 2 + 1}: {file_count}')
        print(f'Files in Folder 0 of Pair {i // 2 + 1}: {file_count_0}, Files in Folder 1 of Pair {i // 2 + 1}: {file_count_1}\n')
    print(f'Total files in all Folder Pairs: {total_files}')


# funtion to preprocess the input images before concatenating them into np arrays, customizable
def preprocessing(image):
    # pixel normalization
    image = np.uint16(image * 255.0)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
    image = clahe.apply(image) + 30
    return image


# funtion to select complementrary images of different modalities but same patient id and slice index, preprocess them,
# and finally concatenate them into numpy arrays
def array_stacking(folder_path, subset=''):
    image_pairs = {}
    stacked_arrays = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for img_path in files:
            img_name = os.path.basename(img_path)
            patient_id, image_type, label, slice_index = parse_image_filename(img_name)
            key = (patient_id, slice_index)
            image_pairs.setdefault(key, {})[image_type] = os.path.join(root, img_path)
            image_pairs[key]['label'] = label
    # Pair and stack images
    for key, image_data in image_pairs.items():
        patient_id, slice_index = key
        if 't2w' in image_data and 'adc' in image_data and 'hbv' in image_data:
            t2w_path = image_data['t2w']
            adc_path = image_data['adc']
            hbv_path = image_data['hbv']
            label = image_data['label']
            # Load and preprocess the images as grayscale
            t2w_image = cv2.imread(t2w_path, cv2.IMREAD_GRAYSCALE)
            adc_image = cv2.imread(adc_path, cv2.IMREAD_GRAYSCALE)
            hbv_image = cv2.imread(hbv_path, cv2.IMREAD_GRAYSCALE)
            t2w_image = preprocessing(t2w_image)
            adc_image = preprocessing(adc_image)
            hbv_image = preprocessing(hbv_image)
            # possibility to apply data augmentation on taining images
            if subset == 'train':
                t2w_array = np.array(t2w_image)
                adc_array = np.array(adc_image)
                hbv_array = np.array(hbv_image)
                t2w_array = np.expand_dims(t2w_image, axis=-1)
                adc_array = np.expand_dims(adc_image, axis=-1)
                hbv_array = np.expand_dims(hbv_image, axis=-1)
                stacked_arrays.append(np.concatenate([t2w_array, adc_array,hbv_array], axis=-1))
                labels.append(label)
            else:
                t2w_array = np.array(t2w_image)
                adc_array = np.array(adc_image)
                hbv_array = np.array(hbv_image)
                t2w_array = np.expand_dims(t2w_image, axis=-1)
                adc_array = np.expand_dims(adc_image, axis=-1)
                hbv_array = np.expand_dims(hbv_image, axis=-1)
                stacked_arrays.append(np.concatenate([t2w_array, adc_array,hbv_array], axis=-1))
                labels.append(label)
    return np.array(stacked_arrays), np.array(labels)


# Custom Weighted Binary Cross-Entropy Loss function
def wbce(weight1, weight0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0)
        return K.mean(logloss, axis=-1)
    return loss


# funtion wich implements learning rate scheduler to use with SGD optimization technique
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch % 50 == 0:
        lr *= 0.1
        return lr
    else:
        return lr