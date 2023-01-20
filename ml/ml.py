import numpy as np
import pandas
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from sklearn.utils import resample

# todo: find way that br removal doesn't glue two words together
from bs4 import BeautifulSoup

from preprocessing import remove_embedded

label_hierarchy = ["technology", "process", "property", "existence", "not-ak"]


def get_highest_tag(tags):
    for label in label_hierarchy:
        if label in tags:
            if label == "technology" or label == "process":
                return "executive"
            return label

    raise Exception("Invalid tag: nog in hierarchy")


def preprocess(raw: pd.DataFrame):
    ret = raw[["SUBJECT", "BODY", "TAGS"]]
    ret["CONTENT"] = ret["SUBJECT"] + " " + ret["BODY"].transform(
        lambda x: remove_embedded(BeautifulSoup(x).get_text()))

    ret["LABEL"] = ret["TAGS"].transform(lambda x: get_highest_tag(x[1:-1].split(", ")))

    return ret[["CONTENT", "LABEL"]]


def extract_features(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()


def evaluate_model(classifier, x_test, y_true):
    y_pred = []
    for feature in x_test:
        y_pred.append(classifier.predict(feature)[0])

    return (
        precision_score(y_true, y_pred, average="weighted"),
        recall_score(y_true, y_pred, average="weighted"),
        f1_score(y_true, y_pred, average="weighted")
    )
    pass


def train(classifier_type, features, labels):
    classifier = classifier_type(force_alpha=True)
    classifier.fit(features, labels)
    return classifier


def batch_train(features, labels, initial_size, increase_step, subset_count, test_size):
    """

    :param features: total 2D array of features
    :param labels: 1D array of labels
    :param initial_size: starting size of a subset
    :param increase_step: how big a subset should differ compared from previous subset
                (e.g. if 100 and subset.len = 200, the next subset will contain 300 features)
    :param subset_count: amount of subsets to create
    :param test_size: either int, stating subset size, or float, stating ratio of data to be training
    :return:
    """

    x, x_test, y, y_true = train_test_split(features, labels, test_size=test_size)

    print(f"Test set:\n{y_true}")

    columns = ["training size", "precision", "recall", "f1"]
    rows = []

    for i in range(subset_count):
        subset_size = initial_size + increase_step * i

        x_train, y_train = resample(x, y, n_samples=subset_size)
        model = train(ComplementNB, x_train, y_train)
        precision, recall, f1 = evaluate_model(model, x_test, y_true)

        rows.append([subset_size, precision, recall, f1])
    return pandas.DataFrame(np.array(rows), columns=columns)


def debug(data):
    preprocessed = preprocess(data)
    features, vocabulary = extract_features(preprocessed["CONTENT"])
    labels = preprocessed["LABEL"]

    initial_size = 10
    increase_step = 10
    subset_count = 5
    test_size = 50
    results = batch_train(features, labels, initial_size, increase_step, subset_count, test_size)

    print(results)


def main():
    df = pd.read_csv("./input/test10.csv")
    debug(data=df)


if __name__ == '__main__':
    main()
