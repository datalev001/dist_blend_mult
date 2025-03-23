import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp

# load data
df = pd.read_csv("marketdata.csv")

# Using Control group for training, Treatment group for testing
train_df = df[df['PromotionFlag'] == 0]
test_df = df[df['PromotionFlag'] == 1]

# Preprocessing
ohe = OneHotEncoder()
channel_train = ohe.fit_transform(train_df[['PreferredChannel']]).toarray()
channel_test = ohe.transform(test_df[['PreferredChannel']]).toarray()

# Properly concatenate and align indices
X_train = pd.concat([
    train_df[['Age','Income','DaysSinceLastPurchase','IsHolidaySeason','LoyaltyScore']].reset_index(drop=True),
    pd.DataFrame(channel_train).reset_index(drop=True)
], axis=1)
y_train = train_df['Purchase'].values

X_test = pd.concat([
    test_df[['Age','Income','DaysSinceLastPurchase','IsHolidaySeason','LoyaltyScore']].reset_index(drop=True),
    pd.DataFrame(channel_test).reset_index(drop=True)
], axis=1)
y_test = test_df['Purchase'].values

# Ensure columns are strings (for scaling)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Feature Scaling
scaler = StandardScaler()

# Convert column names to string explicitly
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Teacher Models
xgb = XGBClassifier().fit(X_tr, y_tr)
lgb = LGBMClassifier().fit(X_tr, y_tr)
rf = RandomForestClassifier().fit(X_tr, y_tr)

def teacher_preds(X):
    return np.vstack([
        xgb.predict_proba(X)[:,1],
        lgb.predict_proba(X)[:,1],
        rf.predict_proba(X)[:,1]
    ]).T

models = {'XGBoost': xgb, 'LightGBM': lgb, 'RandomForest': rf}

for name, model in models.items():
    pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, pred_proba > 0.5)
    ks_stat, _ = ks_2samp(pred_proba[y_test == 1], pred_proba[y_test == 0])
    
    print(f"{name} Performance:")
    print(f"  - AUC: {auc:.4f}")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - KS Statistic: {ks_stat:.4f}\n")

tp_train, tp_val, tp_test = teacher_preds(X_tr), teacher_preds(X_val), teacher_preds(X_test)

# PyTorch Data
X_tr_t, y_tr_t = torch.tensor(X_tr,dtype=torch.float32), torch.tensor(y_tr,dtype=torch.float32).unsqueeze(1)
tp_train_t = torch.tensor(tp_train,dtype=torch.float32)

X_val_t, y_val_t = torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val,dtype=torch.float32).unsqueeze(1)
tp_val_t = torch.tensor(tp_val,dtype=torch.float32)

X_test_t = torch.tensor(X_test,dtype=torch.float32)

# Distillation Model
class AdaptiveDistillModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.student_head = nn.Sequential(nn.Linear(16,1),nn.Sigmoid())
        self.attention_head = nn.Sequential(nn.Linear(16,3),nn.Softmax(dim=1))
    def forward(self,x):
        f=self.shared(x)
        return self.student_head(f),self.attention_head(f)

model=AdaptiveDistillModel(X_tr.shape[1])
optimizer=optim.Adam(model.parameters(),lr=0.0005)
bce_loss=nn.BCELoss()

best_auc, patience=0,5
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    student_pred, att_w = model(X_tr_t)
    teacher_loss = torch.mean(torch.sum(att_w*(tp_train_t-student_pred)**2,dim=1))
    loss = bce_loss(student_pred,y_tr_t) + teacher_loss
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred,_=model(X_val_t)
        val_auc=roc_auc_score(y_val,val_pred.numpy())
        if val_auc>best_auc:
            best_auc=val_auc
            patience=5
            torch.save(model.state_dict(),'best_model.pt')
        else:
            patience-=1
            if patience==0: break
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f}, Val AUC {val_auc:.4f}")

model.load_state_dict(torch.load('best_model.pt'))

# Evaluation
model.eval()
with torch.no_grad():
    test_pred,att_w=model(X_test_t)
    test_pred_np=test_pred.numpy()
    auc=roc_auc_score(y_test,test_pred_np)
    acc=accuracy_score(y_test,test_pred_np>0.5)
    ks_stat,_=ks_2samp(test_pred_np[y_test==1],test_pred_np[y_test==0])
    att_mean=att_w.numpy().mean(axis=0)

print(f"\nTest AUC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test KS: {sum(ks_stat):.4f}")
print(f"Mean Attention Weights (XGB,LGB,RF): {att_mean}")