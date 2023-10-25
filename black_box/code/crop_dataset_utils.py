import numpy as np
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# function to save a chekpoint for the cropping operation, in order not to lose the progress in case
# of runtine errors or discottections
def save_checkpoint(counter, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        f.write(str(counter))
        print(f"Checkpoint saved at folder {counter}")



