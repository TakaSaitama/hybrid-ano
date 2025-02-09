'''
The ingestion program will call `predict` to get a prediction for each test image and then save the predictions for scoring. The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
'''
from open_clip import create_model
from torchvision import transforms
import torch
import pickle
import os
import sys
import pandas as pd
import numpy as np

from transformers import pipeline
from PIL import Image



# Utilities
IMG_SIZE = 256*3
def resize_image(image):
    wpercent = (IMG_SIZE / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image2 = image.resize((IMG_SIZE, hsize), Image.Resampling.LANCZOS)
    return image2
    
def is_overlap(A, B):
    # A = [xmin,ymin,xmax,ymax]
    # B = [xmin,ymin,xmax,ymax]
    rs = not (A[0] > B[2] or A[2] < B[0] or A[1] > B[3] or A[3] < B[1])
    return rs

def does_contain(pp, A, verbose=False):
    for B in pp:
        if is_overlap(A, B):
            if verbose:
                print("Overlap:", B)
            return True
    return False
    
def get_butterfly_wings(image, predictions, return_wings=True, max_nb_wings=4):
    bboxes = []
    counter = 0
    
    sub_images = []
    for prediction in predictions[:max_nb_wings]:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]
    
        xmin, ymin, xmax, ymax = box.values()
        bbox = [xmin, ymin, xmax, ymax]
    
        if not does_contain(bboxes, bbox):
            bboxes.append(bbox)
            
            if return_wings:
                cropped = image.crop((xmin, ymin, xmax, ymax))
                sub_images.append(cropped)
            counter += 1

    if return_wings:
        return bboxes, sub_images
    return bboxes

def get_anomaly_score(cls_predictions):
    df_scores = pd.DataFrame(cls_predictions)
    df_scores["pred"] = cls_predictions.argmax(axis=1)
    df_scores["prob"] = cls_predictions.max(axis=1)
    df_scores = df_scores[df_scores["pred"].isin([1,2,3])]
    score = 0 if (len(df_scores) <= 1) else float(df_scores["prob"].std())
    return float(score)

def get_bin_score(cls_predictions):
    score = cls_predictions[:,1].mean()
    return float(score)

def get_final_score(cls5_predictions, clfbin_predictions, threshold=0.25):
    scorebin = get_bin_score(clfbin_predictions)
    final_score = scorebin
    if scorebin < threshold:
        score5 = get_anomaly_score(cls5_predictions)
        final_score = threshold * score5
    return final_score
    
class Model:
    def __init__(self):
        # model will be called from the load() method
        self.clf = None
        
        try:
            current_dir = os.path.dirname(__file__)
        except:
            current_dir = "./"
        sys.path.insert(1, current_dir)

    def load(self):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        # Features
        model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
        self.model = model.to(self.device)

        self.preprocess_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )

        # Object detection
        checkpoint = "google/owlv2-base-patch16-ensemble"
        self.detector = pipeline(model=checkpoint, task="zero-shot-object-detection", use_fast=True)
       
        # Custom models
        with open(os.path.join(os.path.dirname(__file__), "butterfly_models_5cls.pickle"), "rb") as f:
            self.butterfly_models = pickle.load(f)
            self.clf5 = self.butterfly_models["model"]
        with open(os.path.join(os.path.dirname(__file__), "butterfly_models_bin.pickle"), "rb") as f:
            self.butterfly_models = pickle.load(f)
            self.clfbin = self.butterfly_models["model"]


    def predict(self, datapoint):
        
        with torch.no_grad():
            small_image = resize_image(datapoint)
            predictions = self.detector(
                small_image,
                candidate_labels=["butterfly"], # ["butterfly wing"],
            )
            bboxes, sub_images = get_butterfly_wings(small_image, predictions)

            input_images = [self.preprocess_img(i).to(self.device) for i in sub_images]
            
            score = 0
            if len(input_images) > 0:
            
                image_features = self.model(torch.stack(input_images))['image_features']
                image_features = image_features.detach().cpu().numpy()

                cls5_predictions = self.clf5.predict_proba(image_features)
                clfbin_predictions = self.clfbin.predict_proba(image_features)
                
                score = get_final_score(cls5_predictions, clfbin_predictions)
        return score
