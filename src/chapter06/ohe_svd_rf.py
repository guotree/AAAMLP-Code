import os
import joblib
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

import config
import pandas as pd
def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    features = [f for f in df.columns if f not in ["id", "target", "kfold"]]
    for col in features:
        df.loc[:,col] = df[col].astype(str).fillna("NONE")
    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    ohe = OneHotEncoder()
    full_data = pd.concat([df_train, df_valid], axis=0)
    ohe.fit(full_data[features])
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    svd = TruncatedSVD(n_components=120)
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    model = RandomForestClassifier(n_jobs=-1)
    model.fit(x_train, df_train.target.values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc =  metrics.roc_auc_score(df_valid.target.values, valid_preds)
    print(f"Fold = {fold}, AUC = {auc}")

    if not os.path.exists(config.MODEL_FILE):
        os.makedirs(config.MODEL_FILE)
    joblib.dump(model, os.path.join(config.MODEL_FILE, "ohe_model_{}.pkl".format(fold)))

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)