import itertools

import pandas as pd

from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import preprocessing

def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1+"_"+c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df

def run(fold):
    df = pd.read_csv("../../input/adult_folds.csv")
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    # df.drop(num_cols, axis=1, inplace=True)
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:,"income"] = df.income.map(target_mapping)
    cat_cols = [f for f in df.columns if f not in num_cols and f not in ["kfold", "income"]]
    df = feature_engineering(df, cat_cols)
    features = [
        f for f in df.columns if f not in ["kfold", "income"]
    ]
    for col in features:
        if col not in num_cols:
            df.loc[:,col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col])
            df.loc[:,col] = lbl.transform(df[col])

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values
    model = XGBClassifier(n_jobs=-1, max_depth=7, use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, df_train["income"].values)
    valid_preds = model.predict_proba(x_valid)[:,1]
    auc = metrics.roc_auc_score(df_valid.income, valid_preds)
    print("Fold = {}, AUC = {}".format(fold, auc))
if __name__ == "__main__":
    for fold in range(5):
        run(fold)