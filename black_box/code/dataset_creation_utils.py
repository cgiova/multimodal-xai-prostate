import os
from PIL import Image

# funtion to create the actual dataset from the mapping dataframes, divided by class type
def dataset_creation(dataframe, destination_path:str):
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
        destination_path = os.path.join(clinically_relevant_folder if label == 1 else not_clinically_relevant_folder, image_filename_with_class)
        image = Image.open(image_path)
        # Save the converted image to the destination folder
        image.save(destination_path)
        processed_images += 1
        if processed_images % 50 == 0:
            print(f'Processed {processed_images} images out of {total_images}.')
            progress_updates += 1
    print('Dataset creation completed.')