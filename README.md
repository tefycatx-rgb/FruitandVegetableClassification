# Fruit and Vegetable Classification
This project builds a classification model to predict whether an image contains a fruit or a vegetable.

## Setup Instructions

### 1. Clone this repository

```
git clone https://github.com/natalie8210/FruitandVegetableClassification.git
cd FruitandVegetableClassification
```

### 2. Download training data

Download the dataset from the course SharePoint: 

https://iowa-my.sharepoint.com/personal/washor_uiowa_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwashor%5Fuiowa%5Fedu%2FDocuments%2FSTAT%5F7400%5FImage%5FSubmission&ct=1773247314226&or=Teams%2DHL&ga=1&LOF=1

Steps:
1. Select all 19 folders
2. Click download (589.4 MB)
3. Unzip the downloaded file
4. Move the 19 folders into
```
data/raw/
```
### Important data fix 

Folder 01570623 contains images inside a nested jpeg/ folder.
Fix this by:
* deleting .heic files
* moving .jpeg files into the main 01570623/ folder.

### Final Structure 

```
data/raw/
    01570623/
        img1.jpeg
        ...
    01234567/
        img1.jpg
        ...
```
Each folder should contain 20 images

### 3. Create Virtual Environment
```
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
```
### 4. Install Dependencies
```
pip install -r requirements.txt
```

## Training Workflow (for reference)
Feature Extraction handles by 
```
scripts/preprocess_utils.py
```
Specifically:
```
extract_features_from_image(image_path)
```
* Each image -> 94 features
* 12 images fail preprocessing -> final training set = 368 images

Lables provided in:
```
labels/fruit_labels_metadata.csv
```

## Running Predictions (Instructor Use) 
Complete steps 1-4 above
Copy code snippet from homework into submission.py

### Command
```
python scripts/submission.py \
    --raw_dir <path_to_images> \
    [--output_csv <path_to_output_csv>]
```
## Input Directory Requirements
The --raw_dir must contain images of the following format (images inside subfolder)
```
data/testing/
    batch1/
        image1.jpg
        image2.png
```
## Output Format
The output CSV will contain exactly:
```
image_id,prediction
```
Where:
* image_id = filename without extension
* prediction = fruit or vegetable

### Example
```
python scripts/submission.py \
    --raw_dir data/testing \
    --output_csv results/test_predictions.csv
```
Example output:
```
image_id,prediction
001,fruit
002,vegetable
```

## QC Script
To inspect image quality before running predictions:
```
python scripts/qc_status.py --raw_dir <path_to_images>
```
See:
```
scripts/README.md
```
for more details. 

## Notes
* Supported image formats: .jpg, .jpeg, .png
* The script automatically handles preprocessing, training, and prediction
* Any images that fail preprocessing will recieve a fallback prediction. 

