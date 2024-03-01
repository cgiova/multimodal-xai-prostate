import cv2
import numpy as np
import os
from keras import backend as K
from crop_data_utils import parse_image_filename
from keras.backend import clip, log, mean
from keras.layers import (Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense,LeakyReLU,
                          Dropout, MaxPooling2D)
from keras.regularizers import l2
from keras.models import Model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


#
def count_files_in_folders(*folder_paths: str):
    """
    Function to examine the dataset and check for issues in files
    :param folder_paths: filepath to folders to analyze
    :return: total files number in folder paths
    """
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


def wbce(weight1: int, weight0: int):
    """
    Function for custom Weighted Binary Cross-Entropy Loss
    :param weight1: weight for first parameter
    :param weight0: weight for second parameter
    :return: the loss function
    """""
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0)
        return K.mean(logloss, axis=-1)
    return loss

def scheduler(epoch: int, lr= 1e-3):
    """
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    :param epoch: The number of epochs
    :param lr: the initial learning rate
    :return lr (float32): the learning rate
    """
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#
def scheduler_alt(epoch: int, lr: int, lr_decay=0.1):
    """
    function which implements an alternative learning rate scheduler to use with SGD optimization technique
    :param lr_decay:
    :param epoch: at which epoch or step epoch we want to schedule the learning rate change
    :param lr: the initial learning rate
    :param lr_decay: decay
    :return:
    """
    if epoch < 50:
        return lr
    elif epoch % 50 == 0:
        lr *= lr_decay
        return lr
    else:
        return lr

def residual_block(x, filters, stride=1, first_block=False):
    """
    Function for the Residual Block
    :param x:
    :param filters:
    :param stride:
    :param first_block:
    :return:
    """
    weight_decay=1e-4

    # Shortcut connection
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    # Adjusting dimensions of the shortcut to match the main path
    if first_block:
        shortcut = Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same', kernel_regularizer=l2(weight_decay))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Adding shortcut to the main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def resnet_v2_20(input_shape=input_shape, num_classes=1):
    """
    Function for ResNet-V2 model architecture
    :param input_shape: shape of the input images
    :param num_classes: number of classification labels
    :return: the model
    """
    inputs = Input(shape=input_shape)
    weight_decay=1e-4

    # Initial convolution layer
    x = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Stacking residual blocks
    num_blocks = [3, 3, 3]
    filters = [16, 32, 64]

    for stage in range(3):
        for block in range(num_blocks[stage]):
            stride = 1
            if stage > 0 and block == 0:
                stride = 2

            x = residual_block(x, filters[stage], stride=stride, first_block=(block == 0))

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers for classification
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, x, name='resnet_v2_20')
    return model

def cnn(input_shape=input_shape, num_classes=1):
    """
    Function for a simple CNN architecture
    :param input_shape: shape of the input images
    :param num_classes: number of classification labels
    :return: the model
    """
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    # Global average pooling and dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model



def plot_roc_auc(test_data,y_test,valid_data,y_valid):
    """
    Function plotting the ROC curve and the AUC for the validation set and test set
    :param y_valid:
    :param y_test:
    :param test_data: test data
    :param valid_data: validation data
    :return: the plotted ROC curve
    """
    # Make predictions on the test and validation sets
    predicted_probs_te = model.predict(test_data)
    predicted_probs_val = model.predict(valid_data)

    # Calculate ROC curve and AUC for test set
    fpr_te, tpr_te, thresholds_te = roc_curve(y_test, predicted_probs_te)
    roc_auc_te = auc(fpr_te, tpr_te)

    # Calculate ROC curve and AUC for validation set
    fpr_val, tpr_val, thresholds_val = roc_curve(y_valid, predicted_probs_val)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot ROC curve for both test and validation sets
    plt.figure(figsize=(8, 6))

    plt.plot(fpr_te, tpr_te, color='darkorange', lw=2, label=f'Test (AUC = {roc_auc_te:.2f})')
    plt.plot(fpr_val, tpr_val, color='navy', lw=2, label=f'Validation (AUC = {roc_auc_val:.2f})')

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def show_classification_report(test_data,y_test,valid_data,y_valid):
    """
    :param y_valid:
    :param y_test:
    :param test_data: test data
    :param valid_data: validation data
    :return:
    """
    data_list = [(valid_data, y_valid), (test_data, y_test)]

    for data in data_list:

        # Predict on the dataset using the model
        predictions = model.predict(data[0])
        predicted_labels = np.round(predictions).astype(int)

        # Generate the classification report
        report = classification_report(data[1], predicted_labels, target_names=['Class 0', 'Class 1'])

        if data == data_list[0]:
            print('\nValidation\n', report)
        else:
            print('\nTest\n', report)

def plot_confusion_matrix(y_test):
    """
    Function plotting the confusion matrix for the test set
    :param y_test:
    :return:
    """
    # Make predictions on the test set
    predicted_proba = model.predict(y_test)
    predicted_labels = (predicted_proba > 0.5).astype(int)  # Assuming binary classification with a threshold of 0.5

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_labels)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def balance_dataset(X, y, method='oversample'):
    """
    Balances the dataset by either undersampling or oversampling.
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Target array.
    method (str): Method to balance dataset ('undersample' or 'oversample').
    Returns:
    X_resampled (numpy.ndarray): Resampled feature matrix.
    y_resampled (numpy.ndarray): Resampled target array.
    """
    # Store the original sample dimension for later use
    sample_dimension = X.shape[1:]
    # Flatten X for resampling
    X = X.reshape(X.shape[0], -1)

    unique, counts = np.unique(y, return_counts=True)
    print("Original dataset shape:", dict(zip(unique, counts)))

    if method == 'oversample':
        resampler = RandomOverSampler(random_state=0)
    elif method == 'undersample':
        resampler = RandomUnderSampler(random_state=0)
    else:
        raise ValueError("Method must be 'oversample' or 'undersample'")

    X_resampled, y_resampled = resampler.fit_resample(X, y)
    # Reshape X_resampled to have the original sample dimensions
    X_resampled = X_resampled.reshape(-1, *sample_dimension)

    unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
    print("Resampled dataset shape:", dict(zip(unique_resampled, counts_resampled)))

    return X_resampled, y_resampled