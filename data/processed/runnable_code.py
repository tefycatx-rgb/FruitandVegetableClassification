from preprocess_utils import extract_features_from_image
import os
import pandas as pd
import numpy as np
import joblib 

def run_inference(folder_path, output_csv="predictions.csv"):

    rows = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)
        image_id = os.path.splitext(file)[0]

        try:
            feature_vector, best_mask, best_score, best_params = extract_features_from_image(image_path)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            continue

        # Convert features
        if isinstance(feature_vector, dict):
            feat_dict = feature_vector
        else:
            feat_dict = {f"feat_{i}": v for i, v in enumerate(feature_vector)}

        row = {"image_id": image_id}
        row.update(feat_dict)

        # Include params if needed
        if isinstance(best_params, dict):
            row.update(best_params)

        rows.append(row)

    # Create DataFrame
    features = pd.DataFrame(rows)
    features = features.fillna(features.median(numeric_only=True))

    predictors_selected = [
        "feat_53","feat_89","feat_62","feat_72","feat_64","feat_30",
        "feat_18","feat_14","feat_20","feat_88","feat_67","feat_13",
        "feat_83","feat_27","hole_area","feat_11","feat_41","feat_12",
        "feat_8","feat_39","feat_34","feat_16","feat_17","feat_45",
        "feat_70","feat_23","closing_radius","feat_33","feat_24",
        "feat_40","feat_25","feat_31","feat_32","feat_21"
    ]

    # Handle missing columns safely
    for col in predictors_selected:
        if col not in features:
            features[col] = 0

    X_feat = features[predictors_selected]

    # LOAD TRAINED MODEL
    clf = joblib.load("trained_model.pkl")

    preds = clf.predict(X_feat)

    # Convert predictions
    results = []
    for i, pred in enumerate(preds):
        label = "fruit" if pred else "vegetable"
        results.append({
            "image_id": features.iloc[i]["image_id"],
            "prediction": label
        })

    # Save CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

#run_inference("C:/Users/tefyc/Downloads/live_project_images/01460705")

if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]
    run_inference(folder_path)

