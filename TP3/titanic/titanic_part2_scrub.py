import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from skrub import tabular_pipeline



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare = {"Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"}
    df["Title"] = df["Title"].replace(list(rare), "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # family size & isolation
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # ticket frequency
    df["TicketFreq"] = df["Ticket"].map(df["Ticket"].value_counts())

    # has cabin
    df["Deck"] = df["Cabin"].str[0].fillna("Unknown")

    return df



train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")

train = add_features(train_raw)
test  = add_features(test_raw)


# We include raw Name and Ticket: skrub's StringEncoder will extract useful character n-grams (e.g. "van", "III", "Miss" in the name).

features = [
    # numerical
    "Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone", "TicketFreq",
    # categorical
    "Sex", "Embarked", "Title", "Deck",
    # text (high-cardinality categorical)
    "Name", "Ticket",
]

X      = train[features]
y      = train["Survived"]
X_test = test[features]



model = tabular_pipeline(
    HistGradientBoostingClassifier(
        max_iter=500,
        max_leaf_nodes=24,
        min_samples_leaf=8,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42,
    )
)

cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_raw["PassengerId"],
    "Survived":    predictions,
})
submission.to_csv("submission.csv", index=False)