from crop_data_utils import parse_image_filename2
import numpy as np
import cv2
import os
import pickle as pk


def create_imagepairs_dict(path: str):
    """
    Create a dictionary object with image pairs (patient_id,label):image_type
    Used for stacking the correct images together in the array_stacking function
    """
    image_pairs = {}
    for root, dirs, files in os.walk(path):
        for img_path in files:
            img_name = os.path.basename(img_path)
            patient_id, study_id, image_type, label, slice_index = parse_image_filename2(img_name, class_label=True)
            key = (patient_id, slice_index)
            image_pairs.setdefault(key, {})[image_type] = os.path.join(root,img_path)
            image_pairs[key]['label'] = label

    return image_pairs


def patient_image_map(filepath: str):
    """
    Opens a pickle file and return the mapping dictionary used in
    the array_stacking function
    """
    with open(filepath, 'rb') as fp:
        mapping_dict = pk.load(fp)
    return mapping_dict


def patient_3dimage_map(pickle_file_path: str):
    '''
    Function opening the pickle file obtained from the IGTD algorithm
    the file contains the original tabula data, the 3d array of the images generated
    and the list of samples.
    The function then maps the images to their relative patient_id
    returning a dictionary with the mapping with keys = patient_id ; values = image
    '''
    objects = []
    with (open(pickle_file_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pk.load(openfile))
            except EOFError:
                break

    # original_tab_data = objects[0]
    generated_img_data = objects[1]
    sample_names = objects[2]
    mapping_dict = {}

    for i in range(len(sample_names)):
        patient_id = sample_names[i]
        tabular_image_data = generated_img_data[:, :, i]
        mapping_dict[patient_id] = tabular_image_data

    return mapping_dict


def preprocessing(image):
    """
    Function preprocessing the images with normalization and CLAHE
    :param image: the image to be preprocessed
    :return: the preprocessed image
    """
    image = np.uint16(image * 255.0)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    image = clahe.apply(image) + 30

    return image


def array_stacking(folder_path: str, subset: str, dims: int, mapping_dict_filepath=None):
    """
    Function creating the stacked arrays from the image modalities with or without tabular data
    arrays are patient-wise and slice-wise for images, only patient-wise for tabular data turned into images
    returns a numpy array for each patient slice, the modalities are stacked on the third dimension
    """
    if dims == 4 and mapping_dict_filepath is not None:
        stacked_arrays, labels = array_stacking_4d(folder_path=folder_path,
                                                   mapping_dict_filepath=mapping_dict_filepath,
                                                   subset=subset)
        return stacked_arrays, labels
    else:
        stacked_arrays = []
        labels = []
        lost_counter = 0
        image_pairs = create_imagepairs_dict(folder_path)

        # Pair and stack images
        for key, image_data in image_pairs.items():
            patient_id, slice_index = key
            if 't2w' in image_data and 'adc' in image_data and 'hbv' in image_data:
                t2w_path = image_data['t2w']
                adc_path = image_data['adc']
                hbv_path = image_data['hbv']
                label = image_data['label']

                # Load and preprocess the images as grayscale
                t2w_array = np.array(cv2.imread(t2w_path, cv2.IMREAD_GRAYSCALE))
                adc_array = np.array(cv2.imread(adc_path, cv2.IMREAD_GRAYSCALE))
                hbv_array = np.array(cv2.imread(hbv_path, cv2.IMREAD_GRAYSCALE))

                # possibility to apply data augmentation on training images
                if subset == 'train':
                    t2w_array = np.expand_dims(preprocessing(t2w_array), axis=-1)
                    adc_array = np.expand_dims(preprocessing(adc_array), axis=-1)
                    hbv_array = np.expand_dims(preprocessing(hbv_array), axis=-1)
                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array], axis=-1))
                    labels.append(label)
                else:
                    t2w_array = np.expand_dims(t2w_array, axis=-1)
                    adc_array = np.expand_dims(adc_array, axis=-1)
                    hbv_array = np.expand_dims(hbv_array, axis=-1)
                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array], axis=-1))
                    labels.append(label)
            else:
                lost_counter += 1
                print(f"{patient_id} not found in mapping dict for subset: {subset}")
        print(f"total lost files:{lost_counter}")
        return np.array(stacked_arrays), np.array(labels)


def array_stacking_4d(folder_path: str, mapping_dict_filepath: str, subset: str):
    """
    Function creating the stacked arrays from the other modalities and the tabular data
    arrays are patient-wise and slice-wise for images, only patient-wise for tabular data turned into images
    returns a numpy array for each patient slice, the modalities are stacked on the third dimension
    """
    stacked_arrays = []
    labels = []
    lost_counter = 0
    mapping_dict = patient_image_map(mapping_dict_filepath)
    image_pairs = create_imagepairs_dict(folder_path)

    # Pair and stack images
    for key, image_data in image_pairs.items():
        patient_id, slice_index = key
        if 't2w' in image_data and 'adc' in image_data and 'hbv' in image_data:
            if patient_id in mapping_dict:
                # check if patient is in mapping dictionary
                t2w_path = image_data['t2w']
                adc_path = image_data['adc']
                hbv_path = image_data['hbv']
                label = image_data['label']

                # Load and preprocess the images as grayscale
                t2w_array = np.array(cv2.imread(t2w_path, cv2.IMREAD_GRAYSCALE))
                adc_array = np.array(cv2.imread(adc_path, cv2.IMREAD_GRAYSCALE))
                hbv_array = np.array(cv2.imread(hbv_path, cv2.IMREAD_GRAYSCALE))

                # access the tabular image data of the specific patient
                image_tabular = np.array(mapping_dict[patient_id])

                # Check if the tabular array is empty
                if not image_tabular.size:
                    lost_counter += 1
                    print(f"Tabular array for {patient_id} is empty. Skipping.")
                    continue

                # possibility to apply data augmentation on training images
                if subset == 'train':
                    t2w_array = np.expand_dims(preprocessing(t2w_array), axis=-1)
                    adc_array = np.expand_dims(preprocessing(adc_array), axis=-1)
                    hbv_array = np.expand_dims(preprocessing(hbv_array), axis=-1)
                    tab_img_array = np.expand_dims(image_tabular, axis=-1)
                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array, tab_img_array], axis=-1))
                    labels.append(label)
                else:
                    t2w_array = np.expand_dims(t2w_array, axis=-1)
                    adc_array = np.expand_dims(adc_array, axis=-1)
                    hbv_array = np.expand_dims(hbv_array, axis=-1)
                    tab_img_array = np.expand_dims(image_tabular, axis=-1)
                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array, tab_img_array], axis=-1))
                    labels.append(label)
        else:
            lost_counter += 1
            print(f"{patient_id} not found in mapping dict for subset: {subset}")
    print(f"total lost files:{lost_counter}")
    return np.array(stacked_arrays), np.array(labels)
