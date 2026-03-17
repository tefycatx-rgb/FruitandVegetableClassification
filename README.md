# Fruit and Vegetable Classification

1. Clone this repository 
2. Go to the project sharepoint cite

https://iowa-my.sharepoint.com/personal/washor_uiowa_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwashor%5Fuiowa%5Fedu%2FDocuments%2FSTAT%5F7400%5FImage%5FSubmission&ct=1773247314226&or=Teams%2DHL&ga=1&LOF=1

2. Select all 19 folders, and click download (will take a few minutes, 589.4 MB)
3. The download will be a zipped file. Unzip the file, and move the 19 folders inside to the data/raw folder in your cloned repo
4. Note folder 01570623 has their files within a folder labeled jpeg, so you will need to delete the heic files and move the 20 jpeg files into the main folder. Your data/raw folder should include 19 subfolders, each with 20 images inside that are .jpg, .jpeg, or .png
5. Create a venv
6. Run pip install -r requirements.txt
7. Use scripts/preprocess_utils.py extract_features_from_image function to preprocess each photo in each foler, and save the output in data/preprocess.
     * Note the pipeline failed on 12 images, so the final training set will have 368 items. 
8. Use the preprocessed data, 368 items with 94 features, and fruit_labels_metadata for binary and multiclass labels, for your classification.

For Testing
1. complete steps 1-6 above
2. upload images into data/testing/subfolder
3. run python scripts/qc_status.py --raw_dir data/testing
4. run submissions 

