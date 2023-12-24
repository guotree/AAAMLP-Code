import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from tqdm import tqdm

def create_dataset(training_df, image_dir):
    images = []
    targets = []
    for index, row in tqdm(
        training_df.iterrows(),
        total=len(training_df),
        desc="processing images"
    ):
        image_id = row["imageId"]
        image_path = os.path.join(image_dir, image_id)
        image = Image.open(image_path + ".png")
        image = np.array(image)
        image = image.ravel()
        images.append(image)
        targets.append(int(row["target"]))
    images = np.array(images)
    print(images.shape)
    return images, targets

if __name__ == "__main__":
    csv_path = "/kaggle/input/siim-png-train-csv/train.csv"
    image_path = "/kaggle/input/siim-png-images/train_png/"
    df = pd.read_csv(csv_path)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        xtrain, ytrain = create_dataset(train_df, image_path)
        xtest, ytest = create_dataset(test_df, image_path)
        clf = ensemble.RandomForestClassifier(n_jobs=-1)
        clf.fit(xtrain, ytrain)
        preds = clf.predict_proba(xtest)[:, 1]
        print(f"FOLD: {fold_}")
        print(f"AUC = {metrics.roc_auc_score(ytest, preds)}")
        print("")