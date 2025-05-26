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

import nibabel as nib
import numpy as np
from skimage.transform import resize
import os

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

    # Calculate average cumulative sum density
    avg_density = np.mean(cumsum_slices)

    return top_slice_indices.tolist(), cumsum_slices, avg_density


def process_patient_data(data_path, num_slices,
                         min_density, threshold):
    data_dict = {}
    counter = 0
    density_counter = 0
    tot_avg_density = 0

    for i, patient_file in enumerate(sorted(os.listdir(data_path))):

        if threshold is not None and i >= (threshold - 1):
            break

        if patient_file.endswith('.md'):
            continue

        patient_path = os.path.join(data_path, patient_file)
        patient_id = patient_file.split('.')[0]
        top_slices, cumsum_slices, avg_density = get_crop_coordinates(patient_path, num_slices)
        counter += 1

        print(f'patient num {counter} ({patient_id}) average density: {avg_density}')

        if np.sum(cumsum_slices) > min_density:
            tot_avg_density += avg_density
            density_counter += 1
            if patient_id not in data_dict:
                data_dict[patient_id] = []

            data_dict[patient_id].extend(top_slices)

    print(f'total avg density {tot_avg_density/density_counter:.2f} over {density_counter} patients')
    return data_dict