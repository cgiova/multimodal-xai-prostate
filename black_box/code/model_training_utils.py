import re
import cv2
import numpy as np
import os
from keras import backend as K
from crop_data_utils import parse_image_filename
from keras.backend import clip, log, mean

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

