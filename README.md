# KAMP_casting_optimization_competition
KAMP Competition Prescriptor Modeling repository

## 🚀 Quick Start
```
# clone project
https://github.com/muk-jjang/KAMP_casting_optimization_competition.git

# [Optional] create conda virtual environment
conda create -n myenv python=3.9
conda activate myenv

# install requirements
pip install -r requirements.txt

# run main.ipynb 

```

## 1. 문제 정의
### 1.1 Pain Point
**불량품 Issue** 
- 설비 자체를 경험에 의존하여 제어하기 때문에 품질 불량 문제가 자주 발생함
- 정확한 불량 원인이 불명확함
- 불량품으로 인한 원재료 폐기량이 증가함

**반응고 주조법 Issue**
- 개발 초기 활동은 급증했으나, 현재는 생산량이 매우 적음
- 반응고 주조법에는 공정 조건에 대한 높은 수준의 설비 제어가 필요

### 1.2 최적화 제어 솔루션 제안
- 디지털 트윈을 활용한 최적화 제어 솔루션 제안

## 2. 모델링
### 2.1 디지털 트윈 모델
- XGB
- LightGBM
- EXT(Extra Trees Classifier)
- FTTransformer

### 2.2 최적화 알고리즘
- Baysian Optimization

## 3. 접근 방법
1. 데이터 전처리
2. 디지털 트윈 모델 학습
3. 최적화 알고리즘을 이용하여 불량률을 최소화하는 제어값 탐색
4. 최적화 제어 솔루션 제안


## 4. 후속 대처 방안
- 디지털 트윈 모델을 Continuoual Learning을 통해 지속적으로 업데이트
- 최적화 알고리즘을 통해 탐색한 제어값을 실제 설비에 적용

## Project Structure
```
root
|
├── casting
│   ├── __pycache__             <- compiled python folder
│   │
│   ├── configuration           <- configuration folder
│   │
│   ├── data                    <- data folder
│   │   ├── external data           <- external data folder 
│   │   ├── processed_data          <- processed data folder
│   │   ├── raw_data                <- raw data folder
│   │   ├── scaled_data             <- scaled data folder
│   │   ├── scaler                  <- scaler folder
│   │
│   ├── model                   <- DL model folder
│   │
│   ├── optimizer               <- Bayesian Optimizer    
│   │
│   ├── preprocessing           <- preprocessing folder
│   │   ├── __pycache__             <- compiled python folder
│   │   ├── __init__.py             <- init file
│   │   ├── _data_transform.py      <- data transform file(label_econding, save_scaler)
│   │   ├── _eda.py                 <- eda file(defect rate)
│   │   ├── _preprocessing.py       <- preprocessing file(outlier, DBSCAN, missing value)
│   │
│   ├── results                 <- saved result folder 
│   │
│   ├── trainer                 <- trainer folder
│   │   ├── __pycache__             <- compiled python folder
│   │   ├── __init__.py             <- init file
│   │   ├── _train_DL_model.py      <- train DL model file(FTTransformer)
│   │   ├── _train_ML_model.py      <- train ML model file(XGBoost, LightGBM, EXT)
│   │   
│   ├── utils.py                <- utils file(data loader)
│
├── EDA.ipynb
│
├── main.ipynb 
│
├── requirements.txt
└── README.md
```