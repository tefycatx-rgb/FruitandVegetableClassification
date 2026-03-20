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

#print(vch2.info)
#print(vch2.rsquare)

representatives = (
    vch2.rsquare
        .sort_values(by="RS_Ratio", ascending=False)
        .groupby("Cluster")
        .first()[["Variable", "RS_Ratio"]]
)

#print(representatives)

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
# Accuracy normal Random Forest: 82.67%
# CM [26 0] 
# [13 36]


# Try tuning the model
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=7400),
                    param_grid,
                    cv=5)

grid.fit(X_train_sel, y_train)
best_model = grid.best_estimator_

y_pred_grid = best_model.predict(X_test_sel)
# Accuracy Grid: 78.67%
# CM = [24 2] 
# [14 35]

print("Accuracy_grid:", accuracy_score(y_test, y_pred_grid))
print(confusion_matrix(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

# Try xgboost

from xgboost import XGBClassifier

# Initialize model
xgb = XGBClassifier(
    random_state=7400,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train
xgb.fit(X_train_sel, y_train)

# Predict
preds_xgb = xgb.predict(X_test_sel)

print("Accuracy_xg:", accuracy_score(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))
print(classification_report(y_test, preds_xgb))

accuracy_xgb = accuracy_score(y_test, preds_xgb)
print(f"Accuracy: {accuracy_xgb*100:.2f}%")

# ACC: 0.84
# CM = [23 3]
# [9 40]

misclassified = test_df[y_test != preds_xgb]
print(misclassified)

output = []
for i, pred in enumerate(preds_xgb):
    label = "fruit" if pred else "vegetable"
    output.append({
        "image_id": test_df.iloc[i]["image_id"],
        "prediction": label
    })

pd.DataFrame(output).to_csv("predictions1.csv", index=False)



# Log model:
from sklearn.linear_model import LogisticRegression

# Initialize model
logreg = LogisticRegression(
    random_state=7400,
    max_iter=1000
)

# Train
logreg.fit(X_train_sel, y_train)

# Predict
preds_logreg = logreg.predict(X_test_sel)

print("Accuracy_log:", accuracy_score(y_test, preds_logreg))
print(confusion_matrix(y_test, preds_logreg))
print(classification_report(y_test, preds_logreg))

# ACC: 77.33
# CM = [21 5]
# [12 37]

# SVM
from sklearn.svm import SVC

svm = SVC(
    probability=True,  # important for ensemble later
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=7400
)

svm.fit(X_train_sel, y_train)
preds_svm = svm.predict(X_test_sel)

print("Accuracy_SVM:", accuracy_score(y_test, preds_svm))
print(confusion_matrix(y_test, preds_svm))
print(classification_report(y_test, preds_svm))

# ACC: 62.67
# CM = [14 12]
# [16 33]

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(
    n_estimators=200,
    random_state=7400
)

et.fit(X_train_sel, y_train)
preds_et = et.predict(X_test_sel)

print("Accuracy_ET:", accuracy_score(y_test, preds_et))
print(confusion_matrix(y_test, preds_et))
print(classification_report(y_test, preds_et))

# ACC: 78.67
# CM = [24 2]
# [14 35]

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=7400
)

gb.fit(X_train_sel, y_train)
preds_gb = gb.predict(X_test_sel)
print("Accuracy_gb:", accuracy_score(y_test, preds_gb))
print(confusion_matrix(y_test, preds_gb))
print(classification_report(y_test, preds_gb))

# ACC: 81.33
# CM = [22 4]
# [10 39]

# KNN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)
preds_knn = knn.predict(X_test_scaled)

print("Accuracy_knn:", accuracy_score(y_test, preds_knn))
print(confusion_matrix(y_test, preds_gb))
print(classification_report(y_test, preds_knn))

# ACC: 80
# CM = [22 4]
# [10 39]


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

joblib.dump(xgb, "trained_model.pkl")