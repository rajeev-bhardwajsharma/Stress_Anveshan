from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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


def get_knn():
    return KNeighborsClassifier(
        n_neighbors=2
    )


def get_model(name):
    name = name.lower()

    if name == "rf":
        return get_rf()
    elif name == "svm":
        return get_svm()
    elif name == "knn":
        return get_knn()
    else:
        raise ValueError(f"Unknown model: {name}")