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

        X_train = pd.DataFrame(X_train_num, columns=columns.numeric_columns)
        X_valid = pd.DataFrame(X_valid_num, columns=columns.numeric_columns)
        X_test = pd.DataFrame(X_test_num, columns=columns.numeric_columns)

        X_train[columns.category_columns[0]] = train_cat
        X_valid[columns.category_columns[0]] = valid_cat
        X_test[columns.category_columns[0]] = test_cat

        y_train = train[columns.target_column]
        y_valid = valid[columns.target_column]
        y_test = test[columns.target_column]

    return pd.concat([X_train, y_train], axis=1), pd.concat([X_valid, y_valid], axis=1), pd.concat([X_test, y_test], axis=1)
