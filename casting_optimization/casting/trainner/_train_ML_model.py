
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

from casting.configuration import columns
from sklearn.ensemble import ExtraTreesClassifier


# XGBoost 모델
def train_xgboost(X_train, X_valid, y_train, y_valid):
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    xgb_model.fit(X_train, y_train.values, 
                  eval_set=[(X_train, y_train), (X_valid, y_valid.values)], 
                  early_stopping_rounds=20, 
                  verbose=False)
    return xgb_model

# LightGBM 모델
def train_lightgbm(X_train, X_valid, y_train, y_valid):

    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train.values, 
                  eval_set=[(X_valid, y_valid.values)],
                  )  # 카테고리 변수 추가)
    return lgb_model


def train_extra_trees(X_train, X_valid, y_train, y_valid):
    # Extra Trees 모델 생성
    extra_trees_model = ExtraTreesClassifier(random_state=42,  n_estimators=300)
    # 모델 학습
    extra_trees_model.fit(X_train, y_train)
    
    return extra_trees_model

