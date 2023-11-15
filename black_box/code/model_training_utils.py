import tensorflow as tf
import keras
import re
import cv2
import numpy as np
import os
from keras import backend as K
from keras.backend import clip, log, mean
from keras.callbacks import LearningRateScheduler

def dataset_checker(train_folder_gdrive_0,train_folder_gdrive_1,
                    valid_folder_gdrive_0,valid_folder_gdrive_1,
                    test_folder_gdrive_0,test_folder_gdrive_1):

    _, _, files_train_0 = next(os.walk(train_folder_gdrive_0))
    _, _, files_train_1 = next(os.walk(train_folder_gdrive_1))

    _, _, files_valid_0 = next(os.walk(valid_folder_gdrive_0))
    _, _, files_valid_1 = next(os.walk(valid_folder_gdrive_1))

    _, _, files_test_0 = next(os.walk(test_folder_gdrive_0))
    _, _, files_test_1 = next(os.walk(test_folder_gdrive_1))

    file_count_tr_0 = len(files_train_0)
    file_count_tr_1 = len(files_train_1)
    file_count_tr = int(file_count_tr_0 + file_count_tr_1)

    file_count_val_0 = len(files_valid_0)
    file_count_val_1 = len(files_valid_1)
    file_count_val = int(file_count_val_0 + file_count_val_1)

    file_count_te_0 = len(files_test_0)
    file_count_te_1 = len(files_test_1)
    file_count_te = int(file_count_te_0 + file_count_te_1)

    print(f'total files in GDrive: {file_count_tr + file_count_te + file_count_val}')
    print(f'tot files in GDrive Training set: {file_count_tr}, of which {file_count_tr_0} label "0" and {file_count_tr_1} label "1"')
    print(f'tot files in GDrive Validation set: {file_count_val}, of which {file_count_val_0} label "0" and {file_count_val_1} label "1"')
    print(f'tot files in GDrive Test set: {file_count_te}, of which {file_count_te_0} label "0" and {file_count_te_1} label "1"')

def parse_image_filename(filename):
    #filename = filepath.split('/')[-1]
    parts = filename.split('_')
    patient_id = '_'.join(parts[:2])
    study_id = parts[1]
    image_type = parts[3]
    class_label = int(re.search(r'class(\d+)', parts[-1]).group(1))
    slice_index = int(re.search(rf'{image_type}(\d+)', filename).group(1))
    return patient_id, study_id, image_type, class_label, slice_index

def preprocessing(image):
    image = np.uint16(image * 255.0)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
    image = clahe.apply(image) + 30
    return image

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

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch % 50 == 0:
        lr *= 0.1
        return lr
    else:
        return lr