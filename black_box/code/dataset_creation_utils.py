import os
from PIL import Image
import _pickle as cp
from model_training_utils import parse_image_filename


def dataset_creation(dataframe, destination_path: str):
    """
    Function to create the actual dataset from the mapping pandas dataframe, divided by class type
    :param dataframe:Pandas: pandas dataframe with the label mapping information
    :param destination_path: directory path where the dataset will be saved
    :print: notification when process is completed
    """
    total_images = len(dataframe)
    processed_images = 0
    progress_updates = 0
    # Create the class folders if they don't exist
    not_clinically_relevant_folder = os.path.join(destination_path, '0')
    clinically_relevant_folder = os.path.join(destination_path, '1')
    if not os.path.exists(not_clinically_relevant_folder):
        os.makedirs(not_clinically_relevant_folder)
    if not os.path.exists(clinically_relevant_folder):
        os.makedirs(clinically_relevant_folder)
    # Iterate over each row in the CSV file
    for _, row in dataframe.iterrows():
        image_path = row['image_path']
        label = row['label']
        # Get the image filename
        image_filename = os.path.basename(image_path)
        # Append the class label to the image filename before the file extension
        filename_parts = os.path.splitext(image_filename)
        image_filename_with_class = f'{filename_parts[0]}_class{label}{filename_parts[1]}'
        # Construct the new image path in the destination folder
        destination_path = os.path.join(clinically_relevant_folder if label == 1 else not_clinically_relevant_folder,
                                        image_filename_with_class)
        image = Image.open(image_path)
        # Save the converted image to the destination folder
        image.save(destination_path)
        processed_images += 1
        if processed_images % 50 == 0:
            print(f'Processed {processed_images} images out of {total_images}.')
            progress_updates += 1
    print('Dataset creation completed.')


def patient_3dimage_map(pickle_file_path: str):
    """
    Function opening the pickle file obtained from the IGTD algorithm
    the file containts the original tabula data, the 3d array of the images generated
    and the list of samples.
    :param pickle_file_path:pickle: path to the pickle dump where the results of the tabular data conversion are stored
    :return: dictionary with the mapping with {keys = patient_id ; values = 3d_image}
    """
    objects = []
    with (open(pickle_file_path, "rb")) as openfile:
        while True:
            try:
                objects.append(cp.load(openfile))
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
    Function to preprocess the input images before concatenating them into np arrays, customizable
    :param image: image to preprocess
    :return: image preprocessed
    """
    # pixel normalization
    image = np.uint16(image * 255.0)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    image = clahe.apply(image) + 30
    return image


def array_stacking(folder_path: str, subset=''):
    """
    Function creating the stacked arrays from the 3 modalities
    arrays are patient-wise and slice-wise for images
    returns a numpy array for each patient slice, the modalities are stacked on the third dimension
    :param: folder_path: path to the folder containing the image dataset
    :param: subset: specify the subset to be used (allows for data augmentation and preprocessing to the train set)
    :return: tuple containing the stacked arrays and ground truth labels for training
    """
    image_pairs = {}
    stacked_arrays = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for img_path in files:
            img_name = os.path.basename(img_path)
            patient_id, study_id, image_type, label, slice_index = parse_image_filename(img_name, class_label=True)
            key = (patient_id, slice_index)
            image_pairs.setdefault(key, {})[image_type] = os.path.join(root, img_path)
            image_pairs[key]['label'] = label

    # Pair and stack images
    for key, image_data in image_pairs.items():
        _, slice_index = key
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

            # possibility to apply data augmentation on training images
            if subset == 'train':
                t2w_array = np.expand_dims(np.array(t2w_image), axis=-1)
                adc_array = np.expand_dims(np.array(adc_image), axis=-1)
                hbv_array = np.expand_dims(np.array(hbv_image), axis=-1)
                stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array], axis=-1))
                labels.append(label)

            else:
                t2w_array = np.expand_dims(np.array(t2w_image), axis=-1)
                adc_array = np.expand_dims(np.array(adc_image), axis=-1)
                hbv_array = np.expand_dims(np.array(hbv_image), axis=-1)
                stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array], axis=-1))
                labels.append(label)

    return np.array(stacked_arrays), np.array(labels)


def array_stacking_tabular(folder_path: str, pickle_file_path: str, subset: str):
    """
    Function creating the stacked arrays from the 3 modalities and the tabular data
    arrays are patient-wise and slice-wise for images, only patient-wise for tabular data turned into images
    it return a numpy array for each patient slice, the modalities are stacked on the third dimension
    :param: folder_path: path to the folder containing the image dataset
    :param: pickle_file_path: path to the pickle dump where the results of the tabular data conversion are stored
    :param: subset: specify the subset to be used (allows for data augmentation and preprocessing to the train set)
    :return: tuple containing the stacked arrays and ground truth labels for training
    """
    image_pairs = {}
    stacked_arrays = []
    labels = []
    mapping_dict = patient_3dimage_map(pickle_file_path)
    lost_counter = 0
    for root, dirs, files in os.walk(folder_path):
        for img_path in files:
            img_name = os.path.basename(img_path)
            patient_id, study_id, image_type, label, slice_index = parse_image_filename(img_name, class_label=True)
            key = (patient_id, slice_index)
            image_pairs.setdefault(key, {})[image_type] = os.path.join(root, img_path)
            image_pairs[key]['label'] = label

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
                t2w_image = cv2.imread(t2w_path, cv2.IMREAD_GRAYSCALE)
                adc_image = cv2.imread(adc_path, cv2.IMREAD_GRAYSCALE)
                hbv_image = cv2.imread(hbv_path, cv2.IMREAD_GRAYSCALE)

                # access the tabular image data of the specific patient
                image_tabular = mapping_dict[patient_id]
                # possibility to apply data augmentation on training images
                if subset == 'train':
                    t2w_image = preprocessing(t2w_image)
                    adc_image = preprocessing(adc_image)
                    hbv_image = preprocessing(hbv_image)

                    t2w_array = np.expand_dims(np.array(t2w_image), axis=-1)
                    adc_array = np.expand_dims(np.array(adc_image), axis=-1)
                    hbv_array = np.expand_dims(np.array(hbv_image), axis=-1)
                    tab_img_array = np.expand_dims(image_tabular, axis=-1)

                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array, tab_img_array], axis=-1))
                    labels.append(label)

                else:
                    t2w_array = np.expand_dims(np.array(t2w_image), axis=-1)
                    adc_array = np.expand_dims(np.array(adc_image), axis=-1)
                    hbv_array = np.expand_dims(np.array(hbv_image), axis=-1)
                    tab_img_array = np.expand_dims(image_tabular, axis=-1)

                    stacked_arrays.append(np.concatenate([t2w_array, adc_array, hbv_array, tab_img_array], axis=-1))
                    labels.append(label)
        else:
            lost_counter += 1
            print(f"{patient_id} not found in mapping dict")

    print(f"total lost files:{lost_counter}")
    return np.array(stacked_arrays), np.array(labels)
