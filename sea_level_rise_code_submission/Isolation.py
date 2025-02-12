from sklearn.ensemble import IsolationForest
from GCForest import gcForest
import numpy as np
np.bool = bool

# noinspection PyUnboundLocalVariable
class IsolationModel(object):

    def __init__(self, max_samples=100, random_state=2024, max_inverted_score = 0.5):
        self.model = IsolationForest(max_samples=max_samples, random_state=random_state)
        self.max_inverted_score = max_inverted_score
        
    def fit(self, X_train, y_train=None):
        self.model.fit(X_train)

    def predict_proba(self, X_test):
        y_pred = self.model.decision_function(X_test)
        y_pred = self.max_inverted_score - y_pred
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