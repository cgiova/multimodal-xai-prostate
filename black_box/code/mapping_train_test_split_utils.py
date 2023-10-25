from sklearn.model_selection import train_test_split

def custom_train_test_split(df, test_proportion=None, validation=True, random_state=None):
    unique_patient_ids = df['patient_id'].unique()
    total_samples = len(unique_patient_ids)

    if validation:
        # Calculate the test size based on the specified proportion
        test_proportion = min(test_proportion, 0.5)  # Ensure test proportion is not greater than 0.5

        test_size = int(test_proportion * total_samples)
        validation_size = int(test_proportion * total_samples)

        # Split patient_ids into train, validation, and test sets
        train_patient_ids, remaining_patient_ids = train_test_split(unique_patient_ids, test_size=test_size * 2,
                                                                   random_state=random_state)
        valid_patient_ids, test_patient_ids = train_test_split(remaining_patient_ids, test_size=validation_size,
                                                              random_state=random_state)

        # Filter the dataframe based on the selected patient_ids
        train_df = df[df['patient_id'].isin(train_patient_ids)]
        valid_df = df[df['patient_id'].isin(valid_patient_ids)]
        test_df = df[df['patient_id'].isin(test_patient_ids)]

        return train_df, valid_df, test_df
    else:
        # Calculate the test size based on the specified proportion
        test_proportion = min(test_proportion, 0.5)  # Ensure test proportion is not greater than 0.5

        test_size = int(test_proportion * total_samples)

        # Randomly split patient_ids into train and test sets
        train_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=test_size * 2,
                                                               random_state=random_state)

        # Filter the dataframe based on the selected patient_ids
        train_df = df[df['patient_id'].isin(train_patient_ids)]
        test_df = df[df['patient_id'].isin(test_patient_ids)]

        return train_df, test_df
    

def resize_and_copy_images(source_path, destination_path, new_size=(80, 80)):
    img = Image.open(source_path)
    # img = img.resize(new_size, Image.LANCZOS) # to use if image needs to be resized
    img.save(destination_path)