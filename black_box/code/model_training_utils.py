import re

def parse_image_filename(filename):
    #filename = filepath.split('/')[-1]
    parts = filename.split('_')
    patient_id = '_'.join(parts[:2])
    image_type = parts[3]
    class_label = int(re.search(r'class(\d+)', parts[-1]).group(1))
    slice_index = int(re.search(rf'{image_type}(\d+)', filename).group(1))

    return patient_id, image_type, class_label, slice_index

# Example usage
filepath = '/content/drive/MyDrive/ABELE_prostate/claudio/CSV/dataset/prostate_centered/80x80_stk/rgb/train/0/11100_1001000_adc17_adc_class1.png'
filename = '10000_1000000_adc12_adc_class0.png'

patient_id, image_type, class_label, slice_index = parse_image_filename(filename)
print("Patient ID:", patient_id)
print("Image Type:", image_type)
print("Class Label:", class_label)
print("Slice Index:", slice_index)



