import re

def parse_image_filename(filename):
    #filename = filepath.split('/')[-1]
    parts = filename.split('_')
    patient_id = '_'.join(parts[:2])
    image_type = parts[3]
    class_label = int(re.search(r'class(\d+)', parts[-1]).group(1))
    slice_index = int(re.search(rf'{image_type}(\d+)', filename).group(1))
    return patient_id, image_type, class_label, slice_index



