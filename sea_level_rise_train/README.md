
# Hybrid Solution  

## Overview  
- For each location, we extract SLA values near the coordinates and use a tree-based model to predict anomalies. Instead of using the entire SLA matrix, we focus on relevant subsets.  
- Since there are 12 locations, we train 12 separate submodels.  
- Each model can be either GCForest supervised model or IsolationForest unsupervisied model. The model selection is based on the validation performance metrics.

## Folder structure
## Submission  

$ tree -L 2
.
├── 01-isolation-forest.ipynb
├── 02-hybrid-solution.ipynb
├── GCForest.py
├── input
│   ├── satelite_data
│   └── station_data
├── Isolation.py
├── __pycache__
│   ├── GCForest.cpython-39.pyc
│   ├── Isolation.cpython-39.pyc
│   └── utils.cpython-39.pyc
├── README.md
├── subs
│   └── sub_isolation
└── utils.py




## Train 
- Execute the 2 notebooks 01-isolation-forest.ipynb and 02-hybrid-solution.ipynb

## Submission  
- The code is available in the sea_level_rise_code_submission folder