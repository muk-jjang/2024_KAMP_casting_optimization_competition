from casting.model.pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from pytorch_tabnet.metrics import Metric


def train_tabnet(X_train, X_valid, y_train, y_valid):
    y_train, y_valid = y_train.astype('int'), y_valid.astype('int') 

    y_train = y_train.values.reshape(1, -1)[0]
    y_valid = y_valid.values.reshape(1, -1)[0]

    # criterion=F1ScoreMetric()
    clf = TabNetClassifier()  #TabNetRegressor()
    clf.fit(
    X_train.values, y_train,
    eval_set=[(X_valid.values, y_valid)],
    eval_metric=['auc'],
    patience=20,
    max_epochs=1000,
    )
    return clf

