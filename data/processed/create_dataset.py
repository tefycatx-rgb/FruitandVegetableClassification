import pandas as pd

features = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/features.csv")
other = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/labels/fruit_labels_metadata.csv")

merged = other.merge(features, on="image_id", how="inner")

merged.to_csv("dataset.csv", index=False)


df = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/dataset.csv")

cols_to_drop = ["Unnamed: 0", "student_id", "image_name", "fail_reasons", "qc_status", "best_params"]
# best_params is also eliminated as its data has become 4 independent columns

df = df.drop(columns=cols_to_drop, errors="ignore")

df.to_csv("dataset_clean.csv", index=False)

from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("C:/Users/tefyc/Downloads/FruitandVegetableClassification/dataset_clean.csv")


train_data, test_data = train_test_split(
    data,
    test_size=0.2,      
    random_state=7400,    
    shuffle=True
)

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)