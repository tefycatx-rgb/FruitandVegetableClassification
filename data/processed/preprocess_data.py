from preprocess_utils import load_and_normalize, _corner_samples, segment_fruit, _to_float01, _to_uint8, _apply_mask_preserve_bg, sigf_filter, color_wiener_filter, guided_box_filter, refine_mask, mask_quality_score, optimize_mask, compute_shape_features, circular_mean_std, compute_colour_features, build_feature_vector, visualize_mask, extract_features_from_image

import os
import pandas as pd
import numpy as np


root_folder = "C:/Users/tefyc/Downloads/live_project_images"

rows = []   # list of dicts → becomes DataFrame


for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)
        image_id = os.path.splitext(file)[0]

        try:
            # Try to extract features
            feature_vector, best_mask, best_score, best_params = extract_features_from_image(image_path)

        except Exception as e:
            # If anything fails, skip this image
            print(f"Skipping {file} due to error: {e}")
            continue

        # Convert feature vector to dict
        if isinstance(feature_vector, dict):
            feat_dict = feature_vector
        else:
            feat_dict = {f"feat_{i}": v for i, v in enumerate(feature_vector)}

        # Build row
        row = {
            "image_name": file,
            "image_id": image_id,
            "best_score": best_score,
            "best_params": str(best_params),
        }

        # Expand best_params into columns
        if isinstance(best_params, dict):
            for key, value in best_params.items():
                row[key] = value
        else:
            row["best_params"] = str(best_params)

# Add feature vector columns
        row.update(feat_dict)
        rows.append(row)

# Save results
df = pd.DataFrame(rows)

# Save
df.to_csv("features.csv", index=False)