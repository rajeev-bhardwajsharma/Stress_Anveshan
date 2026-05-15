import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


def load_data():
    base_path = Path(__file__).resolve().parent
    file_path = base_path / "../WDM_dataset/Features/chest/all_chest_statistical.parquet"
    df = pd.read_parquet(file_path)
    return df

def inspect_data(df):
    print("Shape:", df.shape)
    print("\nSubjects:")
    print(df["subject"].value_counts())
    
    print("\nLabels:")
    print(df["label"].value_counts())


def run_loso(df): #using loso
    subjects = df["subject"].unique()

    for test_sub in subjects:
        print(f"\n Testing on {test_sub} ")

        train_df = df[df["subject"] != test_sub]
        test_df  = df[df["subject"] == test_sub]

        X_train = train_df.drop(columns=["label", "subject"])
        y_train = train_df["label"]

        X_test = test_df.drop(columns=["label", "subject"])
        y_test = test_df["label"]

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        accuracy = (preds == y_test).mean()
        print("Accuracy:", accuracy) #have to store it in a text file



def main():
    df = load_data()
    inspect_data(df)
    run_loso(df)


if __name__ == "__main__":
    main()