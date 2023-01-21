import numpy as np
import pandas
import pandas as pd
# todo: find way that br removal doesn't glue two words together
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import ComplementNB
from sklearn.utils import resample

from preprocessing import remove_embedded

label_hierarchy = ["technology", "process", "property", "existence", "not-ak"]


def get_highest_tag(tags):
    """
    Given a list of tags, extract the most important tag
    """

    for label in label_hierarchy:
        if label in tags:
            if label == "technology" or label == "process":
                return "executive"
            return label

    raise Exception("Invalid tag: nog in hierarchy")


def preprocess(raw: pd.DataFrame):
    """
    Preprocess the dataset by
    :param raw:
    :return:
    """
    # get only the columns needed for training
    ret = raw[["SUBJECT", "BODY", "TAGS"]]

    # parse HTML to plain_text and remove embedded threads
    ret["CONTENT"] = ret["SUBJECT"] + " " + ret["BODY"].transform(
        lambda x: remove_embedded(BeautifulSoup(x).get_text()))

    # convert label list to python format and extract the most important one
    ret["LABEL"] = ret["TAGS"].transform(
        lambda x: get_highest_tag(x[1:-1].split(", ")))

    return ret[["CONTENT", "LABEL"]]


def extract_features(corpus):
    """
    Perform feature extraction on a set of strings
    :param corpus: a set of strings, each will be transformed into a feature
    :return: set of features
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()


def evaluate_model(model, x_test, y_true):
    """
    Calculate evaluation metrics for a model
    :param model: the trained model to evaluate
    :param x_test: the features to test on
    :param y_true: ground truth: labels manually assigned
    :return: precision, recall, f1-score
    """
    y_pred = []
    for feature in x_test:
        y_pred.append(model.predict(feature)[0])

    return (
        precision_score(y_true, y_pred, average="weighted"),
        recall_score(y_true, y_pred, average="weighted"),
        f1_score(y_true, y_pred, average="weighted")
    )
    pass


def train(classifier_type, features, labels):
    """
    :param classifier_type: The classifier class to use
    :param features: 2d array: training dataset containing features (e.g. tf-idf)
    :param labels:   1d array respective labels manually given
    :return: the classifier, trained with the respective data
    """
    classifier = classifier_type(force_alpha=True)
    classifier.fit(features, labels)
    return classifier


test_x_sub, test_y_sub = None, None


def batch_train(features, labels, initial_size, increase_step, subset_count):
    """

    :param features: total 2D array of features
    :param labels: 1D array of labels
    :param initial_size: starting size of a subset
    :param increase_step: how big a subset should differ compared from previous subset
                (e.g. if 100 and subset.len = 200, the next subset will contain 300 features)
    :param subset_count: amount of subsets to create
    :return:
    """
    global test_y_sub, test_x_sub  # debugging

    training_times = 5
    kf = KFold(n_splits=training_times)

    # x, x_test, y, y_true = train_test_split(features, labels, test_size=test_size)

    # print(f"Test set:\n{y_true}")

    columns = ["training size", "avg_precision", "avg_recall,", "avg_f1", "precisions", "recalls", "f1s"]
    rows = []

    for i in range(subset_count):
        subset_size = initial_size + increase_step * i
        x_sub, y_sub = resample(features, labels, n_samples=subset_size)
        test_x_sub = x_sub
        test_y_sub = y_sub
        precisions, recalls, f1s = [], [], []

        for train_index, test_index in kf.split(x_sub, y_sub):
            x_train = x_sub[train_index]
            y_train = y_sub[train_index]
            x_test = x_sub[test_index]
            y_true = y_sub[test_index]

            model = train(ComplementNB, x_train, y_train)
            precision, recall, f1 = evaluate_model(model, x_test, y_true)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        rows.append([subset_size,
                     np.mean(precisions),
                     np.mean(recalls),
                     np.mean(f1s),
                     precisions,
                     recalls,
                     f1s
                     ])

    return pandas.DataFrame(rows, columns=columns)


debug_ret = None


def debug(data):
    global debug_ret
    preprocessed = preprocess(data)
    features, vocabulary = extract_features(preprocessed["CONTENT"])
    labels = preprocessed["LABEL"].to_numpy()

    initial_size = 15
    increase_step = 10
    subset_count = 5
    results = batch_train(features, labels, initial_size, increase_step, subset_count)
    debug_ret = results
    print(results)


def main():
    df = pd.read_csv("./input/test10.csv")
    debug(data=df)


if __name__ == '__main__':
    main()
