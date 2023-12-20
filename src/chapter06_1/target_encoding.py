import copy
import os

import joblib
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def mean_target_encoding(data):
    df = copy.deepcopy(data)
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df.loc[:, "income"] = df.income.map(target_mapping)
    features = [f for f in df.columns if f not in ["kfold", "income"] and f not in num_cols]

    for col in features:
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            lbl = LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])
    encoded_dfs = []

    # Target Encoding
    for fold in range(5):
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)
        for column in features:
            mapping_dict = dict(
                df_train.groupby(column)["income"].mean()
            )
            df_valid.loc[:, column + "_enc"] = df_valid[column].map(mapping_dict)
        encoded_dfs.append(df_valid)
    encoded_df = pd.concat(encoded_dfs, axis=0)

    return encoded_df

def run(df, fold, out=0):
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    features = [f for f in df.columns if f not in ["kfold", "income"]]
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    model = XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200, use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, df_train["income"].values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = roc_auc_score(df_valid["income"].values, valid_preds)
    print("Fold = {}, AUC = {}".format(fold, auc))
    model_dir = "../../models/chapter06_1"

    if out==1:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, os.path.join(model_dir+"/model_{}.h5".format(fold)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=int)
    arg = parser.parse_args()
    df = pd.read_csv("../../input/adult_folds.csv")
    df = mean_target_encoding(df)
    for fold in range(5):
        run(df, fold, arg.out)