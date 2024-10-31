from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

## category label encoding
def save_label_encoding(df, cols, paths) :

    le = preprocessing.LabelEncoder()

    for col in cols :
        le.fit(df[col])

    with open(paths, 'wb') as file:
        pickle.dump(le, file)


def save_scaler(df, X, y, X_path, y_path):

    X_pred_scaler = StandardScaler()
    y_pred_scaler = StandardScaler()

    X_pred_df = df[X]
    y_pred_df = df[y]
    X_pred_scaler.fit(X_pred_df)
    y_pred_scaler.fit(y_pred_df)
    
    with open(X_path, 'wb') as file:
        pickle.dump(X_pred_scaler, file)

    with open(y_path, 'wb') as file:
        pickle.dump(y_pred_scaler, file)
    
    