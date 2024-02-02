from sklearn.model_selection import train_test_split
import pandas as pd


def custom_train_test_split(dataframe, test_size: int, val_size=None, random_state=None):
    """
    Function used to perform the train test split or the train val test split in a way that there are no images of the
    same patient across multiple sets (to avoid unwanted correlations during modelling and inference). Also,
    the function assures a very similar class distribution across all sets
    :param dataframe: Pandas: pandas dataframe with the label mapping information
    :param test_size: percentage of the dataset to be used for training
    :param val_size: percentage of the dataset to be used for validation
    :param random_state: seed for reproducibility
    :return: depending on the parameters settings, train and test splits or train test and validation splits
    """
    unique_patient_ids = dataframe['patient_id'].unique()
    # Stratified split on patient_ids and corresponding labels
    labels = dataframe.groupby('patient_id')['label'].first()
    # Initial split to create training and remaining sets
    train_patient_ids, remaining_patient_ids, _, _ = train_test_split(
        unique_patient_ids, labels, test_size=test_size, stratify=labels, random_state=random_state)
    if val_size is not None:
        # Directly split remaining patient IDs into validation and test sets
        valid_patient_ids, test_patient_ids, _, _ = train_test_split(
            remaining_patient_ids, labels[remaining_patient_ids], test_size=val_size,
            stratify=labels[remaining_patient_ids],
            random_state=random_state)
        # Filter the dataframe based on the selected patient_ids
        train_df = dataframe[dataframe['patient_id'].isin(train_patient_ids)]
        valid_df = dataframe[dataframe['patient_id'].isin(valid_patient_ids)]
        test_df = dataframe[dataframe['patient_id'].isin(test_patient_ids)]
        return train_df, valid_df, test_df
    else:
        train_df = dataframe[dataframe['patient_id'].isin(train_patient_ids)]
        test_df = dataframe[dataframe['patient_id'].isin(remaining_patient_ids)]
        return train_df, test_df


def check_dataframes(train_df, test_df, valid_df=None):
    """
    Function used to check the newly created dfs with metrics such as size, class distribution, number of overlapping
    patient across sets
    :param train_df:Pandas: training set dataframe
    :param test_df:Pandas: testing set dataframe
    :param valid_df:Pandas: validation set dataframe
    :print: info about the dataframes overlapping data, if any
    """
    if valid_df is not None:
        # check_overlapping
        train_patient_ids = set(train_df['patient_id'].unique())
        valid_patient_ids = set(valid_df['patient_id'].unique())
        test_patient_ids = set(test_df['patient_id'].unique())
        overlapping_train_valid = train_patient_ids.intersection(valid_patient_ids)
        overlapping_train_test = train_patient_ids.intersection(test_patient_ids)
        overlapping_valid_test = valid_patient_ids.intersection(test_patient_ids)
        print('Total size:', len(train_df) + len(valid_df) + len(test_df))
        print('Train dataframe size:', len(train_df))
        print('Validation dataframe size:', len(valid_df))
        print('Test dataframe size:', len(test_df))
        print('Train dataframe unique IDs:', train_df['patient_id'].nunique())
        print('Validation dataframe unique IDs:', valid_df['patient_id'].nunique())
        print('Test dataframe unique IDs:', test_df['patient_id'].nunique(), "\n")
        print('Train dataframe samples:', train_df.head(), "\n")
        print('Validation dataframe samples:', valid_df.head(), "\n")
        print('Test dataframe samples:', test_df.head(), "\n")
        if len(overlapping_train_valid) == 0:
            print("No overlapping patient IDs between train and validation dataframes.")
        else:
            print("Overlapping patient IDs found between train and validation dataframes:")
            print(overlapping_train_valid)
        if len(overlapping_train_test) == 0:
            print("No overlapping patient IDs between train and test dataframes.")
        else:
            print("Overlapping patient IDs found between train and test dataframes:")
            print(overlapping_train_test)
        if len(overlapping_valid_test) == 0:
            print("No overlapping patient IDs between validation and test dataframes.")
        else:
            print("Overlapping patient IDs found between validation and test dataframes:")
            print(overlapping_valid_test)
        # checking class distribution in the dfs
        for i, df in enumerate([train_df, test_df, valid_df]):
            label_counts = df['label'].value_counts().sort_index()
            total_samples = len(df)
            percentage_per_class = (label_counts / total_samples) * 100

            result_df = pd.DataFrame({
                'Count': label_counts,
                'Percentage': percentage_per_class
            }).sort_index()
            print(f"\nClass balance for dataframe {i + 1}:\n")
            print(result_df)
            print("\n" + "=" * 40 + "\n")
    else:
        train_patient_ids = set(train_df['patient_id'].unique())
        test_patient_ids = set(test_df['patient_id'].unique())
        overlapping_train_test = train_patient_ids.intersection(test_patient_ids)
        print('Train dataframe size:', len(train_df))
        print('Test dataframe size:', len(test_df))
        print('Train dataframe unique IDs:', train_df['patient_id'].nunique())
        print('Test dataframe unique IDs:', test_df['patient_id'].nunique(), "\n")
        print('Train dataframe samples:', train_df.head(), "\n")
        print('Test dataframe samples:', test_df.head(), "\n")
        if len(overlapping_train_test) == 0:
            print("No overlapping patient IDs between train and test dataframes.\n")
        else:
            print("Overlapping patient IDs found between train and test dataframes:", overlapping_train_test, "\n")
        for i, df in enumerate([train_df, test_df]):
            label_counts = df['label'].value_counts().sort_index()
            total_samples = len(df)
            percentage_per_class = (label_counts / total_samples) * 100
            result_df = pd.DataFrame({
                'Count': label_counts,
                'Percentage': percentage_per_class
            }).sort_index()
            print(f"Class balance for dataframe {i + 1}:\n")
            print(result_df)
            print("\n" + "=" * 40 + "\n")
