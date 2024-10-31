import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split


def drop_null(df) :
    return df.dropna()

def remove_extreme_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    # 하위와 상위 퍼센트 경계를 설정하고, 그 안에 속하는 값들만 유지
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def split_by_dbscan(data, folder_path):

    # 데이터 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # min_samples를 scaled_data의 컬럼 개수로 설정
    min_samples = scaled_data.shape[1]

    # DBSCAN 클러스터링
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)  # eps와 min_samples는 데이터에 따라 조정 필요
    clusters = dbscan.fit_predict(scaled_data)
    # 데이터에 클러스터 할당
    data = data.assign(Cluster=clusters)

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for c in np.unique(clusters):
        cluster_data = data[data['Cluster'] == c]

        # 클러스터 내 데이터가 충분히 많은지 확인
        if len(cluster_data) > 3:
            X_train, X_temp = train_test_split(cluster_data, test_size=0.3, random_state=42)
            X_val, X_test = train_test_split(X_temp, test_size=1/5, random_state=42)
        else:
            # 클러스터 내 데이터가 3개 이하인 경우의 로직
            X_train, X_val, X_test = cluster_data, pd.DataFrame(), pd.DataFrame()

        # 클러스터별 데이터셋 합치기
        train_data = pd.concat([train_data, X_train])
        valid_data = pd.concat([valid_data, X_val])
        test_data = pd.concat([test_data, X_test])

    # 파일 경로 생성
    base_filename = os.path.basename(folder_path).split('.')[0]
    train_filepath = os.path.join(os.path.dirname(folder_path), f"{base_filename}train.csv")
    valid_filepath = os.path.join(os.path.dirname(folder_path), f"{base_filename}valid.csv")
    test_filepath = os.path.join(os.path.dirname(folder_path), f"{base_filename}test.csv")
    all_filepath = os.path.join(os.path.dirname(folder_path), f"{base_filename}all.csv")

    # CSV 파일로 저장하기 전 'Cluster' 열 제거
    train_data.drop('Cluster', axis=1, inplace=True)
    valid_data.drop('Cluster', axis=1, inplace=True)
    test_data.drop('Cluster', axis=1, inplace=True)

    # CSV 파일로 저장
    train_data.to_csv(train_filepath, index=False)
    valid_data.to_csv(valid_filepath, index=False)
    test_data.to_csv(test_filepath, index=False)

    pd.concat([train_data, valid_data, test_data]).to_csv(all_filepath, index = False)

