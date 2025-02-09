# Hybrid Solution  

## Overview  
- For each location, we extract SLA values near the coordinates and use a tree-based model to predict anomalies. Instead of using the entire SLA matrix, we focus on relevant subsets.  
- Since there are 12 locations, we train 12 separate submodels.  
- Each model can be either GCForest supervised model or IsolationForest unsupervisied model. The model selection is based on the validation performance metrics.

## Running the Model  
- Execute the ingestion script to run the model. The code is available at:  
  [GitHub Repository](https://github.com/iharp-institute/HDR-ML-Challenge-public/blob/main/codabench_files/ingestion_program/ingestion.py)  

## Solution Components  
- **model.py** – The trained model.  
- **dict_meta.pickle** – A checkpoint containing model metadata.  
- **baseline_sla.csv** – A dataset where each row represents a single day. Columns correspond to station locations monitoring anomalies. Binary values (0 for no anomaly, 1 for anomaly) indicate predictions for each station.  

