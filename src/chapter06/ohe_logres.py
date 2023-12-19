import os
import joblib
import argparse

import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import config

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    features = [f for f in df.columns if f not in ("id","target","kfold")]

    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")

    if model == "lr":
        ohe = preprocessing.OneHotEncoder()
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
        ohe.fit(full_data[features])
        x_train = ohe.transform(df_train[features])
        x_valid = ohe.transform(df_valid[features])
    elif model == "rf":
        for col in features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        x_train = df_train[features].values
        x_valid = df_valid[features].values

    m = config.MODELS.get(model)
    m.fit(x_train, df_train["target"].values)
    valid_preds = m.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print("Fold = {:d}, AUC = {:.5f}".format(fold, auc))
    if not os.path.exists(config.MODEL_FILE):
        os.makedirs(config.MODEL_FILE)
    joblib.dump(m, os.path.join(config.MODEL_FILE, "ohe_model_{}.pkl".format(fold)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    for i in range(args.fold+1):
        run(i, args.model)