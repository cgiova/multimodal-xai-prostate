# Multimodal Deep Learning and Explainable AI for Enhanced Prostate Lesion Classification GitHub Repo

Welcome to the repository for the paper "Integrating Multimodal Deep Learning and Explainable AI for Enhanced Prostate Lesion Classification".
This repository contains Python scripts and Jupyter notebooks related to the creation of a black box classifier, dataset preprocessing, and explainability techniques.

## Paper Abstract

Artificial Intelligence holds immense promise in healthcare, offering opportunities to enhance disease diagnosis, treatment selection, and clinical testing. However, the lack of interpretability in AI systems poses a significant challenge, potentially impeding their adoption. In our paper, we address this issue by proposing an analytical framework that leverages a multimodal deep learning approach for the classification of prostate lesions using Magnetic Resonance Imaging (MRI) data and patient clinical information.

We first present a multimodal deep learning model that integrates MRI data and patient metadata to classify prostate lesions. This model aims to improve diagnostic accuracy while maintaining interpretability. Subsequently, we introduce a multimodal explainable artificial intelligence (XAI) approach, employing visual explanations to elucidate the decision-making process of our proposed model. This approach allows us to identify how different modalities contribute to each specific prediction, enhancing trust and understanding.

Our experiments, conducted on the PI-CAI Grand Challenge dataset, demonstrate the potential of combining multimodal data and XAI techniques to advance prostate cancer diagnosis. By integrating MRI data such as Apparent Diffusion Coefficient (ADC), T2-weighted (T2w), and Diffusion-Weighted Imaging (DWI), along with tabular metadata, and employing a convolutional neural network enhanced by transfer learning, our study offers improved predictive performance, reliability, and interpretability in medical diagnostics and treatment decisions.

## Repository Contents

- `black_box/`: Directory containing scripts related to the black box classifier.
  - `code/`: Subdirectory containing additional scripts related to the project.
    - `IGTD_utils.py`: Utility functions for the black box classifier.
    - `array_creation_stacking_utils.py`: Utility functions for creating and stacking arrays.
    - `crop_data_utils.py`: Utility functions for cropping data notebook.
    - `dataset_creation_utils.py`: Utility functions for dataset creation.
    - `mapping_train_test_split_utils.py`: Utility functions for mapping train-test splits.
    - `model_training_utils.py`: Utility functions for training models.
  - `notebooks/`: Subdirectory containing Jupyter notebooks.
    - `0.lesion_EDA.ipynb`: Notebook for exploratory data analysis on prostate lesions.
    - `1.crop_data.ipynb`: Notebook for cropping data preprocessing step.
    - `2.mapping_train_test_split.ipynb`: Notebook for mapping train-test split preprocessing step.
    - `3.1.dataset_creation.ipynb`: Notebook for dataset creation preprocessing step.
    - `3.2.tab_to_images.ipynb`: Notebook for converting tabular data to images.
    - `3.3.array_creation_stacking.ipynb`: Notebook for array creation and stacking preprocessing step.
    - `4.model_training.ipynb`: Notebook for training the classification model.
    - `5.post_lesion_explanation_creation.ipynb`: Notebook for post-lesion explanation analysis.
- `README.md`: This file you're currently reading.

## Citation

If you find our work helpful, please consider citing our paper:

\[citation\]

## Contributors (in alphabetical order) and Organizations
- [Andrea Berti](https://github.com/----)
- [Sara Colantonio](https://github.com/----)
- [Claudio Giovannoni]([https://github.com/johndoe](https://github.com/cgiova))
- [Carlo Metta](https://github.com/----)
- [Anna Monreale](https://github.com/----)
- [Francesca Pratesi](https://github.com/----)
- [Salvatore Rinzivillo](https://github.com/----)
- [University of Pisa]([https://github.com/unipisa]): Organization
- [ISTI-CNR]((https://www.isti.cnr.it/en/)): Organization

---
This repository is maintained by the authors and contributors. We welcome any feedback or suggestions for improvement.
