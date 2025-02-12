from transformers import pipeline
import numpy as np
from PIL import Image
import glob
import random
from tqdm import tqdm
import pickle

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

input_dir = "input/"
filenames = glob.glob(input_dir + "images/**/*.jpg")
print(len(filenames))

dict_rs = {}
for counter, filename in tqdm(enumerate(filenames)):
    image = Image.open(filename)
    predictions = detector(
        image,
        candidate_labels=["butterfly wing"],
    )
    dict_rs[filename] = predictions

    if counter < 5:
        print(counter, filename, len(predictions))

with open('workspace/butterfly_boxes.pickle', 'wb') as handle:
    pickle.dump(dict_rs, handle)

