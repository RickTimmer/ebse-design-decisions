import numpy as np
import pandas
import pandas as pd
# todo: find way that br removal doesn't glue two words together
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from math import ceil
from warnings import simplefilter
import timeit

from formatting import printClassifierLatex, printIterationLatex
from preprocessing import preprocess


def extract_features(corpus, vectorizer):
    """
    Perform feature extraction on a set of strings
    :param corpus: a set of strings, each will be transformed into a feature
    :return: set of features
    """
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
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="macro")
    )


test_x_sub, test_y_sub = None, None


def batch_train(features, labels, classifier, increase_step, kfold_splits):
    """
    :param features: total 2D array of features
    :param labels: 1D array of labels
    :param classifier: The classifier that is to be used
    :param increase_step: how big a subset should differ compared from previous subset
                (e.g. if 100 and subset.len = 200, the next subset will contain 300 features)
    :param kfold_splits: number of splits done for the kfold
    :return:
    """
    global test_y_sub, test_x_sub  # debugging

    kf = KFold(n_splits=kfold_splits)

    columns = ["training size", "avg_precision", "avg_recall,", "avg_f1", "precisions", "recalls", "f1s"]
    rows = []

    # maybe change this, smaller subsets will be trained more than higher ones
    initial_size = increase_step
    subset_count = ceil(features.shape[0] / increase_step)
    leftover_size = features.shape[0] % increase_step

    for i in range(subset_count):
        subset_size = initial_size + increase_step * i
        if i == subset_count - 1 and leftover_size != 0:
            subset_size = subset_size + leftover_size - increase_step
        x_sub, y_sub = resample(features, labels, n_samples=subset_size)

        precisions, recalls, f1s = [], [], []

        for train_index, test_index in kf.split(x_sub, y_sub):
            x_train = x_sub[train_index]
            y_train = y_sub[train_index]
            x_test = x_sub[test_index]
            y_true = y_sub[test_index]

            model = classifier.fit(x_train, y_train)
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
    simplefilter("ignore")
    global debug_ret
    preprocessed = preprocess(data)
    features_tfidf, _ = extract_features(preprocessed["CONTENT"], TfidfVectorizer())
    features_count, _ = extract_features(preprocessed["CONTENT"], CountVectorizer())
    labels = preprocessed["LABEL"].to_numpy()

    increase_step = 100
    kfold_splits = 5
    vectorizers = [
        {"name": "Tfidf", "features": features_tfidf},
        {"name": "Count", "features": features_count},
    ]
    classifiers = [
        {"classifier": ComplementNB(), "name": "Complement Naive Bayes", "short_name": "CNB"},
        {"classifier": DecisionTreeClassifier(), "name": "Decision Tree", "short_name": "DT"},
        {"classifier": RandomForestClassifier(), "name": "Random Forest", "short_name": "RF"},
        {"classifier": LinearSVC(), "name": "Linear Support Vector Classification", "short_name": "LSV"}
    ]
    for classifier in classifiers:
        for vectorizer in vectorizers:
            start = timeit.default_timer()
            results = batch_train(vectorizer["features"], labels, classifier["classifier"], increase_step, kfold_splits)
            stop = timeit.default_timer()
            debug_ret = results

            print("--------------- " + vectorizer["name"] + " -> " + classifier["name"] + " --- " + "Time: " + str(
                stop - start) + " ---------------")
            print(results)
            print("---------------------")

            # This collects the metrics of the latest iteration for each classifier to generate latex bar charts
            classifier[vectorizer["name"] + "precision"] = results.iloc[-1]["avg_precision"]
            classifier[vectorizer["name"] + "recall"] = results.iloc[-1][
                2]  # "avg_recall" for some reason isn't working.
            classifier[vectorizer["name"] + "f1"] = results.iloc[-1]["avg_f1"]

            printIterationLatex(results, vectorizer, classifier)

    printClassifierLatex(classifiers, vectorizers)


def main():
    df = pd.read_csv("./input.csv")
    debug(data=df)


if __name__ == '__main__':
    main()
