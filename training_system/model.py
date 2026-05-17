from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_rf():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )


def get_svm():
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True # for giving probability instead of hardcore 0 or 1 
    )

"""
removing the KNN not needed and adding xgboost model
"""
def get_xgboost():
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

def get_model(name):
    name = name.lower()

    if name == "rf":
        return get_rf()
    elif name == "svm":
        return get_svm()
    elif name == "knn":
        return get_knn()
    elif name=="xgboost":
        return get_xgboost()
    else:
        raise ValueError(f"Unknown model: {name}")