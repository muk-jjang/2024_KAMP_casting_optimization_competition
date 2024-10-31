import pandas as pd
from casting.configuration.config import paths,columns


def load_data(model_kind = 'ml'):
    if model_kind == 'ml':
        train = pd.read_csv(paths.ml_train_path)
        valid = pd.read_csv(paths.ml_valid_path)
        test = pd.read_csv(paths.ml_test_path)
    elif model_kind == 'FTT':
        ''
    
    X_train, y_train = train[columns.input_columns], train[columns.target_column]
    X_valid, y_valid = valid[columns.input_columns], valid[columns.target_column]

    return X_train, y_train, X_valid, y_valid