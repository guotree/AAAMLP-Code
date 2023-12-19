from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

TRAINING_FILE = "../../input/cat_train_folds.csv"
MODEL_FILE = "../../models/chapter06/"
MODELS = {
    "rf": RandomForestClassifier(n_jobs=-1),
    "lr": LogisticRegression(),
}