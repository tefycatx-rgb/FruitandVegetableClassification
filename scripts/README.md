## QC Usage

```
git clone https://github.com/natalie8210/FruitandVegetableClassification.git
```
Note: May need to add '!' to the beginning depending on the environment.

```
%cd FruitandVegetableClassification
python scripts/qc_status.py
```
This outputs the count of flag vs. passed photos. It also creates qc_metrics.csv which give the metrics of the different quality control measures (e.g. exposure, blur, etc.).

<img width="277" height="218" alt="image" src="https://github.com/user-attachments/assets/05f50e24-e0bf-432f-84df-28e0bb19bc68" />

Everyone's raw data will be uploaded to the `data/raw/` directory on GitHub so there should be no need to do anything else but run the above code. 

If you are just wanting to test the functionality of qc_status.py (Prof. Wang's task), then once the repo is cloned, you can upload test data (photos) to the `raw` folder inside whatever environment you are working in. 
