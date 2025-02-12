import pickle
import os
import sys
import pandas as pd
import numpy as np

import logging
import datetime

def get_sla_values(X, verbose=False):
    str_type = str(type(X))
    a = X
    if verbose:
        print(str_type)
    if "xarray.core.dataset.Dataset" in str_type :
        a = X["sla"].values
    if "netCDF4._netCDF4.Dataset" in str_type:
        a = X.variables["sla"]
        a = np.ma.getdata(sla.filled(0))

    a = a[0,:,:]
    a[np.isnan(a)] = 0
    return a
    
cities = [
    "Atlantic City",
    "Baltimore",
    "Eastport",
    "Fort Pulaski",
    "Lewes",
    "New London",
    "Newport",
    "Portland",
    "Sandy Hook",
    "Sewells Point",
    "The Battery",
    "Washington",
]

# 100 160
cities_indexes = {
    'Atlantic City': {'latitude': (55, 60), 'longitude': (100, 105)},
    'Baltimore': {'latitude': (55, 60), 'longitude': (91, 96)},
    'Eastport': {'latitude': (77, 82), 'longitude': (130, 135)},
    'Fort Pulaski': {'latitude': (26, 31), 'longitude': (74, 79)},
    'Lewes': {'latitude': (53, 58), 'longitude': (97, 102)},
    'New London': {'latitude': (63, 68), 'longitude': (109, 114)},
    'Newport': {'latitude': (63, 68), 'longitude': (112, 117)},
    'Portland': {'latitude': (72, 77), 'longitude': (116, 121)},
    'Sandy Hook': {'latitude': (59, 64), 'longitude': (101, 106)},
    'Sewells Point': {'latitude': (45, 50), 'longitude': (92, 97)},
    'The Battery': {'latitude': (60, 65), 'longitude': (101, 106)},
    'Washington': {'latitude': (53, 58), 'longitude': (89, 94)}
}

def get_features_X_row(X_test, city, params, verbose=False):
    sla = X_test # get_sla_values(X)
    indexes = cities_indexes[city]
    latitude = indexes["latitude"]
    longitude = indexes["longitude"]

    margin = params["margin"]

    sla_row = sla[latitude[0] - margin : latitude[1] + margin, longitude[0] - margin : longitude[1] + margin]
    if verbose:
        print(sla.shape, sla_row.shape)
        print(sla.shape)

    return sla_row
    


class Model:
    def __init__(self):
        logging.warning(f"Loading at {datetime.datetime.now()}")

        self.load()

        print("Model initialized")
        logging.warning(f"Loaded at {datetime.datetime.now()}")

    def load(self):
        try:
            current_dir = os.path.dirname(__file__)
        except:
            current_dir = "./"
        logging.warning(f"current_dir is {current_dir}")
        sys.path.insert(1, current_dir)

        # Custom models
        with open(os.path.join(current_dir, "dict_meta.pickle"), "rb") as f:
            self.meta = pickle.load(f)

        self.counter = 0
        self.threshold_ratio = 1

    def get_prediction(self, X):
        X_test = get_sla_values(X)
        y_pred = []
        for city in cities:
            model_run = self.meta[city]
            model = model_run["model"]
            params = model_run["params"]
            best_threshold = params["best_threshold"]
            pred = 0
            
            if best_threshold > 0:
                img = get_features_X_row(X_test, city, params)
                prob = float(model.predict_proba([img.flatten()])[0,1])
                pred = 1 if float(prob) >= (best_threshold*self.threshold_ratio) else 0
            else:
                pred = 0
                
            y_pred.append(pred)

        # print(city, params, prob, pred)
        self.counter += 1
        logging.warning(f"{datetime.datetime.now()} - {self.counter} - {y_pred}")

        return np.array([y_pred])

    def predict(self, X):

        y_pred = self.get_prediction(X)

        return y_pred
        # return np.zeros((1, 12))
        # return np.ones((1, 12))
