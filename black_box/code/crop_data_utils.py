import numpy as np
import nibabel as nib #remember to install nibabel library in the notebook
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tqdm import tqdm
import re
import glob

# funtion to parse and obtain the key elements from the image filenames of the dataset
def parse_image_filename(filename,class_label=True):
    if class_label:
        parts = filename.split('_')
        patient_id = '_'.join(parts[:2])
        study_id = parts[1]
        image_type = parts[3]
        class_label = int(re.search(r'class(\d+)', parts[-1]).group(1))
        slice_index = int(re.search(rf'{image_type}(\d+)', filename).group(1))
        return patient_id, study_id, image_type, class_label, slice_index
    else:
        filename = filename.split('.')[0]
        parts = filename.split('_')
        patient_id = '_'.join(parts[:2])
        study_id = parts[1]
        image_type = parts[-1]
        slice_index = int(parts[2][3:])
        return patient_id, study_id, image_type, slice_index

# funtion to display a selected image
def img_printer(img_path:str):
    img = mpimg.imread(img_path)
    print(f'Image shape: {img.shape}\n')
    plt.imshow(img)
    plt.show()

# function to get the cropping coordinates of the prostate gland trough the annotation file
# this is needed to then crop the prostate image at correct ROI
def get_crop_coordinates(delineation_path, num_slices):
    try:
        img = nib.load(delineation_path)
        img_fdata = img.get_fdata()
    except Exception as e:
        print(f"Error loading NIfTI file: {delineation_path}")
        print(f"Error message: {str(e)}")
        return [], []

    img_resized = resize(img_fdata, (384, 384), preserve_range=True)
    cumsum_slices = np.sum(np.sum(img_resized, axis=0), axis=0)
    # Get indices of top slices based on overall density
    top_slice_indices = np.argsort(cumsum_slices)[-num_slices:][::-1]

    # List to store cropping coordinates
    rois = []
    for slice_idx in top_slice_indices:
        # Process each top slice
        annotation_slice = resize(img_fdata[:, :, slice_idx], (384, 384), preserve_range=True)
        mask = annotation_slice > 0
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:
            continue

        xmin, xmax = np.min(cols), np.max(cols)
        ymin, ymax = np.min(rows), np.max(rows)
        # Calculate the size of the bounding box
        width = xmax - xmin
        height = ymax - ymin
        # Calculate the maximum dimension (width or height) of the bounding box
        max_dimension = max(width, height)
        # Calculate the center of the bounding box
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2
        # Calculate the crop boundaries around the center
        crop_size = max_dimension
        half_crop_size = crop_size // 2
        x_min = max(0, x_center - half_crop_size)
        x_max = min(384, x_center + half_crop_size)
        y_min = max(0, y_center - half_crop_size)
        y_max = min(384, y_center + half_crop_size)
        # Adjust the crop boundaries if the width is zero
        if x_max - x_min == 0:
            x_min = max(0, x_center - half_crop_size - 1)
            x_max = min(384, x_center + half_crop_size + 1)
        if y_max - y_min == 0:
            y_min = max(0, y_center - half_crop_size - 1)
            y_max = min(384, y_center + half_crop_size + 1)
        # Ensure that the crop boundaries are within the image dimensions
        x_min = max(0, x_min)
        x_max = min(384, x_max)
        y_min = max(0, y_min)
        y_max = min(384, y_max)

        # Append the crop coordinates for the current slice to the list
        rois.append([x_min, x_max, y_min, y_max])
        # Print the slice number and relative density
        #print(f"Slice {slice_idx}: Overall Density = {np.sum(annotation_slice)}")

    return rois, top_slice_indices.tolist()

# funtion to control if any of the newly created folders has anomalies i.e. more files than intended
def check_anomalies(output_path:str, threshold:int):
    # Initialize counters for total patient directories with anomalies and total anomalies
    patient_dirs_with_anomalies = 0
    total_anomalies = 0
    total_positive_anomalies = 0
    total_negative_anomalies = 0
    tot_folders = 0
    # Iterate through the patient directories
    for patient_dir in os.listdir(output_path):
        patient_path = os.path.join(output_path, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        tot_folders += 1
        # Flag to track if any subdirectory has anomalies in the patient directory
        has_anomalies_patient = False
        count = 0
        # Count the number of file paths within the subdirectory
        num_files = len([f for f in os.listdir(patient_path) if os.path.isfile(os.path.join(patient_path, f))])
        # Check if the number of file paths exceeds the threshold
        if num_files > threshold:
            count +=1
            has_anomalies_patient = True
            total_anomalies += (num_files - threshold)
            total_positive_anomalies += (num_files - threshold)
        elif num_files < threshold:
            count +=1
            has_anomalies_patient = True
            total_anomalies += (num_files - threshold)
            total_negative_anomalies += (abs(num_files - threshold))
        # Check if any subdirectory in the patient directory has anomalies
        if has_anomalies_patient:
            patient_dirs_with_anomalies += 1
            print(f"Patient directory '{patient_dir}' contains subdirectories with {count} anomalies.")

    print(f"\nTotal number of patient directories: {tot_folders}")
    print(f"Total number of patient directories with anomalies: {patient_dirs_with_anomalies}")
    print(f"Total number of anomalies: {total_anomalies}")
    print(f"Total number of positive anomalies: {total_positive_anomalies}")
    print(f"Total number of negative anomalies: {total_negative_anomalies}")

# funtion to check the number of folders, files and overall gb size of the data directory
def file_checker(path:str):
    if os.path.exists(path):
        total_size = 0
        num_patient_folders = 0
        tot_files = 0
        for dirpath, dirnames, filenames in tqdm(os.walk(path)):
            for dirname in dirnames:
                if dirname.isdigit():
                    num_patient_folders += 1

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
                tot_files += 1

        print(f'\nTotal number of patient folders: {num_patient_folders}')
        print(f'Total number of patient files: {tot_files}')
        print(f"Total size of {path.strip('/')[-1]}: {total_size / (1024 ** 3):.2f} Gigabytes")
        return num_patient_folders
    else:
        print('Specified path does not exist')

# funtion to crop the images and saved them in an output directory based on the coordinates and slices
# obtained with the get_crop_coordinates(delineation_path, num_slices) funtion
# used only for testing purposes, as it lacks a checkpoint logic
def crop_dataset(input_path:str,output_root:str,crop_shape:tuple,num_slices:int, folder_breakpoint:int,
                    wholegland_del_path:str, excluded_patients:list):
    processed_folders = 1
    print('Begin Dataset Creation!',"\n")
    # Loop over the patient folders
    for patient_folder in os.listdir(input_path):
        if processed_folders % 5 == 0:
            print(f"\nprocessing folder number... {processed_folders}: {patient_folder}\n")
        else:
            print(f"\nProcessing folder: {patient_folder} \n")
        if patient_folder in excluded_patients:
            continue
        patient_path = os.path.join(input_path, patient_folder)
        if os.path.isdir(patient_path):
            # Create output directory fo each patient
            output_path = os.path.join(output_root, patient_folder)
            os.makedirs(output_path, exist_ok=True)
            # Loop over the image files in the patient folder
            for image_path in glob.glob(os.path.join(patient_path, '*.png')):
                patient_filename = image_path.split('/')[-1]
                patient_id, _, image_type, slice_num = parse_image_filename(patient_filename, class_label=False)
                print(f"\npatient_filename {patient_filename}")
                print(f"patient_id {patient_id}")
                #print(f"study_id {study_id}")
                print(f"image_type {image_type}")
                print(f"slice_num {slice_num}\n")
                # loop over to get the specific patient path
                wholegland_del_patient_path = os.path.join(wholegland_del_path, patient_id + '.nii.gz')
                try:
                    rois, slices = get_crop_coordinates(wholegland_del_patient_path,num_slices)
                    # check if the slice is in the list of positive delineations
                    if slice_num in slices:
                        # Get the coordinates for the current slice from the rois list
                        coordinates_list = rois[slices.index(slice_num)]
                        if re.search(r'_t2w\d{1,2}_t2w.png', image_path) or re.search(r'_t2w\d{1,2}_adc.png', image_path):
                            t2w_img = mpimg.imread(image_path)
                            t2w_img_crop = t2w_img[coordinates_list[0]:coordinates_list[1],\
                                                    coordinates_list[2]:coordinates_list[3],:]
                            # resize the image to a fixed format
                            t2w_img_resized = resize(t2w_img_crop, crop_shape, preserve_range=True)
                            # extract the image name without extension
                            img_name = os.path.splitext(os.path.basename(image_path))[0]
                            if img_name.endswith("_adc"):
                                new_img_name = img_name.replace("_adc", "_t2w")
                            else:
                                new_img_name = img_name
                            # Save the cropped image with the new filename
                            output_img_path = os.path.join(output_path, new_img_name + '.png')
                            # save the cropped image to the output directory
                            plt.imsave(output_img_path, t2w_img_resized)
                        # same logic for adc and hbv format images
                        elif re.search(r'_adc\d{1,2}_adc.png', image_path):
                            adc_img = mpimg.imread(image_path)
                            adc_img_crop = adc_img[coordinates_list[0]:coordinates_list[1],\
                                                    coordinates_list[2]:coordinates_list[3],:]
                            adc_img_resized = resize(adc_img_crop, crop_shape, preserve_range=True)
                            img_name = os.path.splitext(os.path.basename(image_path))[0]
                            output_img_path = os.path.join(output_path, img_name + '.png')
                            plt.imsave(output_img_path, adc_img_resized)
                        elif re.search(r'_hbv\d{1,2}_hbv.png', image_path):
                            hbv_img = mpimg.imread(image_path)
                            hbv_img_crop = hbv_img[coordinates_list[0]:coordinates_list[1],\
                                                    coordinates_list[2]:coordinates_list[3],:]
                            hbv_img_resized = resize(hbv_img_crop, crop_shape, preserve_range=True)
                            img_name = os.path.splitext(os.path.basename(image_path))[0]
                            output_img_path = os.path.join(output_path, img_name + '.png')
                            plt.imsave(output_img_path, hbv_img_resized)
                except Exception as e:
                    print(f"Error processing slice: {slice_num} in folder: {patient_folder}")
                    print(f"Error message: {str(e)}")
                    continue
        # update counter
        processed_folders += 1
        # Break the loop if the desired number of folders have been processed
        if folder_breakpoint:
            if processed_folders >= folder_breakpoint:
                print("\t\t",'Total folders processed...', processed_folders,"\n")
                break
    print("Dataset Creation Completed!")

# funtion to crop the images and saved them in an output directory based on the coordinates and slices
# obtained with the get_crop_coordinates(delineation_path, num_slices) funtion
# includes a checkpoint system to resume if executions stops
def crop_dataset_checkflag(input_path: str, output_root: str, crop_shape: tuple, num_slices: int,
                            folder_breakpoint: int, wholegland_del_path:str, excluded_patients: list,
                            checkpoint_file: str):
    processed_folders = 1
    print('Begin Dataset Creation!', "\n")
    # check checkpoint
    if checkpoint_file:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint_counter = int(f.read())
                print("\t", f"Checkpoint found! Resuming from folder number... {checkpoint_counter}")
        else:
            checkpoint_counter = 0
            with open(checkpoint_file, "w") as f:
                f.write(str(0))
                print("\t", f"Checkpoint not found. Creating new Checkpoint")
        # Loop over the patient folders
        for patient_folder in os.listdir(input_path):
            if processed_folders < checkpoint_counter:
                processed_folders += 1
                continue
            if patient_folder in excluded_patients:
                continue
            if processed_folders % 1 == 0:
                print(f"\n\t\tProcessing folder number... {processed_folders}: {patient_folder}")
                with open(checkpoint_file,'w') as f:
                    checkpoint_counter = processed_folders
                    f.write(str(checkpoint_counter))
            patient_path = os.path.join(input_path, patient_folder)
            if os.path.isdir(patient_path):
                output_path = os.path.join(output_root, patient_folder)
                os.makedirs(output_path, exist_ok=True)
                for image_path in glob.glob(os.path.join(patient_path, '*.png')):
                    patient_filename = image_path.split('/')[-1]
                    patient_id, _, image_type, slice_num = parse_image_filename(patient_filename, class_label=False)
                    wholegland_del_patient_path = os.path.join(wholegland_del_path, patient_id + '.nii.gz')
                    try:
                        rois, slices = get_crop_coordinates(wholegland_del_patient_path, num_slices)
                        if slice_num in slices:
                            coordinates_list = rois[slices.index(slice_num)]
                            if re.search(r'_t2w\d{1,2}_t2w.png', image_path) or re.search(
                                    r'_t2w\d{1,2}_adc.png', image_path):
                                t2w_img = mpimg.imread(image_path)
                                t2w_img_crop = t2w_img[coordinates_list[0]:coordinates_list[1], \
                                                        coordinates_list[2]:coordinates_list[3], :]
                                t2w_img_resized = resize(t2w_img_crop, crop_shape, preserve_range=True)
                                img_name = os.path.splitext(os.path.basename(image_path))[0]
                                if img_name.endswith("_adc"):
                                    new_img_name = img_name.replace("_adc", "_t2w")
                                else:
                                    new_img_name = img_name
                                output_img_path = os.path.join(output_path, new_img_name + '.png')
                                plt.imsave(output_img_path, t2w_img_resized)
                            elif re.search(r'_adc\d{1,2}_adc.png', image_path):
                                adc_img = mpimg.imread(image_path)
                                adc_img_crop = adc_img[coordinates_list[0]:coordinates_list[1], \
                                                        coordinates_list[2]:coordinates_list[3], :]
                                adc_img_resized = resize(adc_img_crop, crop_shape, preserve_range=True)
                                img_name = os.path.splitext(os.path.basename(image_path))[0]
                                output_img_path = os.path.join(output_path, img_name + '.png')
                                plt.imsave(output_img_path, adc_img_resized)
                            elif re.search(r'_hbv\d{1,2}_hbv.png', image_path):
                                hbv_img = mpimg.imread(image_path)
                                hbv_img_crop = hbv_img[coordinates_list[0]:coordinates_list[1], \
                                                        coordinates_list[2]:coordinates_list[3], :]
                                hbv_img_resized = resize(hbv_img_crop, crop_shape, preserve_range=True)
                                img_name = os.path.splitext(os.path.basename(image_path))[0]
                                output_img_path = os.path.join(output_path, img_name + '.png')
                                plt.imsave(output_img_path, hbv_img_resized)
                    except Exception as e:
                        # print(f"Error processing slice: {slice_num} in folder: {patient_folder}")
                        # print(f"Error message: {str(e)}")
                        continue
                processed_folders += 1
                if folder_breakpoint:
                    if processed_folders >= folder_breakpoint:
                        print("\t\t", 'Total folders processed...', processed_folders, "\n")
                        break
        # Save the final checkpoint
        with open(checkpoint_file, 'w') as f:
            f.write(str(processed_folders))
        print("Dataset Creation Completed!")
    else:
        crop_dataset(input_path, output_root, crop_shape, num_slices, folder_breakpoint, wholegland_del_path,
                        excluded_patients)