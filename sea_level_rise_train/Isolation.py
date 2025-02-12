from sklearn.ensemble import IsolationForest
from GCForest import gcForest
import numpy as np
np.bool = bool

# noinspection PyUnboundLocalVariable
from sklearn.ensemble import IsolationForest
from GCForest import gcForest
import numpy as np
np.bool = bool

# noinspection PyUnboundLocalVariable
class IsolationModel(object):

    def __init__(self, max_samples=100, random_state=2024, max_inverted_score = 0.5):
        self.model = IsolationForest(max_samples=max_samples, random_state=random_state)
        self.max_inverted_score = max_inverted_score

        self.params = {
            "max_inverted_score": max_inverted_score,
            "calibrated": False
        }
        
    def fit(self, X_train, y_train=None, calibration=True):
        self.model.fit(X_train)
        if calibration:
            self.calibrate(X_train)

    def calibrate(self, X_train, y_train=None):
        y_pred = self.model.decision_function(X_train)
        min_y = float(y_pred.min())
        max_y = float(y_pred.max())

        self.params["min_y"] = min_y
        self.params["max_y"] = max_y
        self.params["calibrated"] = True
        

    def predict_proba(self, X_test):
        y_pred = self.model.decision_function(X_test)
        if self.params["calibrated"] == False:
            y_pred = self.max_inverted_score - y_pred
        else:
            y_pred = (y_pred-self.params["min_y"]) / self.params["max_y"]
            y_pred = 1 - y_pred
            
        y_pred = np.clip(y_pred, 0, 1)

        y_0 = 1-y_pred
        y_pred = np.column_stack((y_0, y_pred))
        return y_pred


# from GCForest import gcForest # https://github.com/pylablanche/gcForest/tree/master
from GCForest import gcForest

class GcForestModel(gcForest):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
            
    def predict_proba(self, X_test):
        y_pred = super().predict_proba(X_test)
        y_pred = y_pred[:,-1]
        return y_pred