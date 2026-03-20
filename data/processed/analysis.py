import pandas as pd
from varclushi import VarClusHi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/train.csv")
test = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/test.csv")

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

# Impute data
train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))

cols_to_drop = [
    "image_id",
    "food_name",
    "is_fruit",
    "fruit_instance",
    "lighting_session",
    "background_id",
    "blue_score",
    "exposure_score",
    "blur_score",
    "saturation_score",
    "occupancy_score"
]

y_train = train_df["is_fruit"]
X_train = train_df.drop(columns=cols_to_drop, errors="ignore")

y_test = test_df["is_fruit"]
X_test = test_df.drop(columns=cols_to_drop, errors="ignore")


vch2 = VarClusHi(X_train)
vch2.varclus()

print(vch2.info)
print(vch2.rsquare)

representatives = (
    vch2.rsquare
        .sort_values(by="RS_Ratio", ascending=False)
        .groupby("Cluster")
        .first()[["Variable", "RS_Ratio"]]
)

print(representatives)

selected_vars = representatives["Variable"].tolist()

X_train_sel = X_train[selected_vars]
X_test_sel = X_test[selected_vars]

clf = RandomForestClassifier(random_state=7400)
clf.fit(X_train_sel, y_train)

preds = clf.predict(X_test_sel)

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy*100:.2f}%")

misclassified = test_df[y_test != preds]
print(misclassified)

output = []
for i, pred in enumerate(preds):
    label = "fruit" if pred else "vegetable"
    output.append({
        "image_id": test_df.iloc[i]["image_id"],
        "prediction": label
    })

pd.DataFrame(output).to_csv("predictions1.csv", index=False)


predictors_selected = [
    "feat_53",
    "feat_89",
    "feat_62",
    "feat_72",
    "feat_64",
    "feat_30",
    "feat_18",
    "feat_14",
    "feat_20",
    "feat_88",
    "feat_67",
    "feat_13",
    "feat_83",
    "feat_27",
    "hole_area",
    "feat_11",
    "feat_41",
    "feat_12",
    "feat_8",
    "feat_39",
    "feat_34",
    "feat_16",
    "feat_17",
    "feat_45",
    "feat_70",
    "feat_23",
    "closing_radius",
    "feat_33",
    "feat_24",
    "feat_40",
    "feat_25",
    "feat_31",
    "feat_32",
    "feat_21"
]

import joblib

joblib.dump(clf, "trained_model.pkl")