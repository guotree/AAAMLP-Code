import os

import joblib
import pandas as pd
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

import config


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    features = [f for f in df.columns if f not in ["id","target","kfold"]]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    for col in features:
        lbl = LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])
    df_train = df[df.kfold !=fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = XGBClassifier(n_jobs=-1, max_depth=5, n_estimators=200,use_label_encoder=False,eval_metric='logloss')
    model.fit(x_train, df_train["target"].values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid["target"].values, valid_preds)
    print("Fold {}: AUC: {:.4f}".format(fold, auc))
    if not os.path.exists(config.MODEL_FILE):
        os.makedirs(config.MODEL_FILE)
    joblib.dump(model, os.path.join(config.MODEL_FILE, "ohe_model_{}.pkl".format(fold)))

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)