# Copyright 2025 Claudio Giovannoni, Carlo Metta, Anna Monreale,
# Salvatore Rinzivillo, Andrea Berti, Sara Colantonio, and
# Francesca Pratesi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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