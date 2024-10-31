from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle

## category label encoding
def label_encoding(df, cols) :
    le = preprocessing.LabelEncoder()
    for col in cols :
        df[col] = le.fit_transform(df[col])
    return df

## numeric_data_scainlg
def save_min_max_scale(df, X, y, X_path, y_path):
    X_pred_scaler = MinMaxScaler()
    y_pred_scaler = MinMaxScaler()

    X_pred_df = df[X]
    y_pred_df = df[y]
    X_pred_scaler.fit(X_pred_df)
    y_pred_scaler.fit(y_pred_df)
    
    with open(X_path, 'wb') as file:
        pickle.dump(X_pred_scaler, file)

    with open(y_path, 'wb') as file:
        pickle.dump(y_pred_scaler, file)