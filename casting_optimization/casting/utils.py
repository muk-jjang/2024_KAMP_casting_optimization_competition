import pandas as pd
import numpy as np
from casting.configuration.config import paths,columns
import pickle


def load_data(model_kind = 'ml'):

    if model_kind == 'ml':
        train = pd.read_csv(paths.ml_train_path)
        valid = pd.read_csv(paths.ml_valid_path)
        test = pd.read_csv(paths.ml_test_path)

        X_train, y_train = train[columns.input_columns], train[columns.target_column]
        X_valid, y_valid = valid[columns.input_columns], valid[columns.target_column]
        X_test, y_test = test[columns.input_columns], test[columns.target_column]

    elif model_kind == 'FTT':
        train = pd.read_csv(paths.dl_train_path)
        valid = pd.read_csv(paths.dl_valid_path)
        test = pd.read_csv(paths.dl_test_path)

        # Load the saved scaler
        with open(paths.X_scaler_path, 'rb') as file:
            X_scaler = pickle.load(file)
        with open(paths.label_encoding_path, 'rb') as file:
            encoder = pickle.load(file)

        train_cat = encoder.transform(train[columns.category_columns])
        valid_cat = encoder.transform(valid[columns.category_columns])
        test_cat = encoder.transform(test[columns.category_columns])

        X_train_num = X_scaler.transform(train[columns.numeric_columns])
        X_valid_num = X_scaler.transform(valid[columns.numeric_columns])
        X_test_num = X_scaler.transform(test[columns.numeric_columns])

        X_train = np.hstack((X_train_num, np.array(train_cat).reshape(-1, 1)))
        X_valid = np.hstack((X_valid_num, np.array(valid_cat).reshape(-1, 1)))
        X_test = np.hstack((X_test_num, np.array(test_cat).reshape(-1, 1)))

        y_train = train[columns.target_column].values
        y_valid = valid[columns.target_column].values
        y_test = test[columns.target_column].values

    return X_train, y_train, X_valid, y_valid, X_test, y_test
