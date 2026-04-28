import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def add_features(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Groups vs solo travelers signal: passengers sharing the same ticket are likely to be in a group
    ticket_counts    = df["Ticket"].map(df["Ticket"].value_counts())
    df["TicketFreq"] = ticket_counts.fillna(1).astype(int)


    return df

def add_family_survival(train, test):
    """
    Family survival rate:
    - Family = same last name + similar fare (binned)
    - Leave-one-out on train to avoid label leakage
    - Only use the feature if the family has ≥ 2 known members
    """

    train = train.copy()
    test  = test.copy()

    default = train["Survived"].mean()

    for df in [train, test]:
        df["_LastName"]  = df["Name"].str.split(",").str[0]
        df["_FamilyKey"] = df["_LastName"] + "_" + (df["Fare"].fillna(0) // 5).astype(int).astype(str)

    grp = (
        train.groupby("_FamilyKey")["Survived"]
        .agg(grp_sum="sum", grp_count="count")
    )
    train = train.join(grp, on="_FamilyKey")

    # Leave-one-out strict
    loo_denom = (train["grp_count"] - 1).clip(lower=1)
    loo_rate  = (train["grp_sum"] - train["Survived"]) / loo_denom

    # Use only if at least 1 OTHER member is known
    train["FamilySurvivalRate"] = np.where(
        train["grp_count"] >= 2,
        loo_rate,
        default,
    )

    # Test
    grp_rate = (grp["grp_sum"] / grp["grp_count"]).rename("FamilySurvivalRate")
    test = test.join(grp_rate, on="_FamilyKey")
    test["FamilySurvivalRate"] = test["FamilySurvivalRate"].fillna(default)

    # drop temporary columns
    train.drop(columns=["_LastName","_FamilyKey","grp_sum","grp_count"], inplace=True)
    test.drop(columns=["_LastName","_FamilyKey"], inplace=True)

    return train, test


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_fe = add_features(train)
test_fe = add_features(test)

train_fe, test_fe = add_family_survival(train_fe, test_fe)



numeric_features = ["Age", "Fare", "FamilySize", "TicketFreq", "FamilySurvivalRate"]
categorical_features = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin"]

features = numeric_features + categorical_features

X = train_fe[features]
y = train_fe["Survived"]
X_test = test_fe[features]


numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=2,
                random_state=42,
            ),
        ),


    ]
)

# CROSS VALIDATION
cv_results = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("CV Accuracy: %.4f ± %.4f" % (cv_results.mean(), cv_results.std()))

model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"],
        "Survived": predictions,
    }
)
submission.to_csv("submission.csv", index=False)

