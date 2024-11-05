from casting.model.pytorch_tabnet.tab_model import TabNetClassifier
from casting.model.ft_transformer import FTTransformer    

from casting.configuration.config import columns, paths
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from torch import nn
import numpy as np
import pandas as pd
import random

seed = 2022
deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_tabnet(train, valid):
    X_train, y_train = train[columns.input_columns], train[columns.target_column]
    X_valid, y_valid = valid[columns.input_columns], valid[columns.target_column]
    
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




class FttDataloader(Dataset):
    def __init__(self, data):

        # end inverse-one-hot encoding  

        self.X_numer = np.array(data[columns.numeric_columns], dtype=np.float32)
        self.X_categ = data[columns.category_columns].values.reshape(-1, 1)
        self.label = data[columns.target_column].values.astype(np.float32).reshape(-1, 1)

    def __len__(self): 
        return len(self.X_numer)    
    
    def __getitem__(self, index):
        return self.X_categ[index], self.X_numer[index], self.label[index]      


def train_FTT(train, valid):
        
    train_data = FttDataloader(train)
    valid_data = FttDataloader(valid)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False)

    model = FTTransformer(
    categories = (7,),      # tuple containing the number of unique values within each category
    num_continuous = train_data.X_numer.shape[1],                # number of continuous values
    dim = 64,                           # dimension, paper set at 32, Widedeep set at 64
    dim_out = 1, 
    depth = 6,                          # depth, paper recommended 6, Widedeep set at 4
    heads = 8,                          # heads, paper recommends 8, Widedeep set at 8
    attn_dropout = 0.1,                 # post-attention dropout, Widedeep set at 0.2s
    ff_dropout = 0.1                    # feed forward dropout, Widedeep set at 0.1
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_schedulers= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min",
                                    min_lr=1e-5,
                                    factor=0.5,
                                    verbose=1)
    

    patience = 20  # 얼리 스톱을 위한 최대 기다림 횟수
    es = 0  # 현재 기다림 횟수
    num_epochs = 1000
    loss_min = 9999.0 # 학습 중 최소 loss 기록용

    for epoch in range(num_epochs):
        loss_fn = nn.BCELoss()
        loss_e = 0.0
        loss_count = 0 
        model.train()  # 모델을 훈련 모드로 설정
        for batch, (x_categ, x_numer, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_categ, x_numer, y = x_categ.to(device), x_numer.to(device), y.to(device)
   
            pred = model(x_categ, x_numer)
            loss = loss_fn(pred, y)
            loss_e += loss
            loss_count+=1
            loss.backward()
            optimizer.step()
        loss_e = loss_e / (loss_count)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss_e}')

        # 검증
        loss_e = 0.0
        loss_count = 0
        model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():
            for batch, (x_categ, x_numer, y) in enumerate(valid_loader):
                    x_categ, x_numer, y = x_categ.to(device), x_numer.to(device), y.to(device)
                    pred = model(x_categ, x_numer)
                    pred = pred.cpu().detach().numpy()
                    y = y.cpu().detach().numpy()    
                    # pred = torch.Tensor(label_transformer.inverse_transform(pred))
                    # y = torch.Tensor(label_transformer.inverse_transform(y))
                    loss_e += loss
                    loss_count+=1
            loss_e = loss_e / (loss_count)

            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {loss_e}')

        # 얼리 스톱 체크
        lr_schedulers.step(loss_e)
        if loss_e < loss_min : 
            print(f'{epoch} epoch validation loss : {loss_e}  !!Min_Loss!!')
            loss_min = loss_e
            es = 0
            torch.save(model.state_dict(),paths.FTT_path)
        else:
            es += 1
            if es >= patience:
                print("early_stop!!!: 검증 손실이 향상되지 않아 훈련 종료.")
                break

