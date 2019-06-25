from app import app

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import sklearn.metrics as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_selection import SelectFromModel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets

from pathlib import Path


@app.route("/")
@app.route("/index")
def index():
    return "Hello, World!"


@app.route("/example/")
def example():
    return {"hello": "world"}


@app.route("/classify/<key>")
def classify(key):
    audio_data_good = pd.read_csv("audio_data_good.csv", encoding="latin-1")
    audio_data_bad = pd.read_csv("audio_data_bad.csv", encoding="latin-1")

    audio_data_good["Playlist"] = "Good"

    audio_data_bad["Playlist"] = "Bad"

    audio_data_master = audio_data_good.append(audio_data_bad)

    goodBadClassifer = audio_data_master.loc[:, ["Playlist"]]
    audio_features = audio_data_master.loc[
        :, ["BPM", "ENERGY", "DANCE", "LOUD", "VALENCE", "ACOUSTIC", "POP."]
    ]

    alfred_adalearn = AdaBoostClassifier()
    Xtrain, XTest, yTrain, yTest = train_test_split(
        audio_features, goodBadClassifer, test_size=0.33
    )

    alfred_adalearn.fit(Xtrain, yTrain)
    ypred = alfred_adalearn.predict(XTest)

    return {"key": key, "Alfred acc": accuracy_score(yTest, ypred)}


@app.route("/rebuild/")
def rebuild():
    return {"rebuilding": "OK"}

