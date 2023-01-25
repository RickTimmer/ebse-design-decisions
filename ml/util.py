import os
from dataclasses import dataclass

import pandas as pd
import random

from time import time
from numpy import floor

from preprocessing import preprocess


@dataclass
class Thread:
    title: str
    emails = []


def append_or_add(store, key, val):
    if key in store:
        store[key].append(val)
    else:
        store[key] = [val]


def get_threads(subjects: pd.Series):
    """
    Find at which index a thread starts
    :param subjects:
    :return:
    """
    threads = {}

    for iloc, subject in enumerate(subjects.tolist()):
        if subject.startswith("Re:"):
            subject = subject[3:].strip()
        append_or_add(threads, subject, iloc)

    return threads


def get_labels_per_threads(threads, preprocessed):
    raw = {thread: preprocessed.iloc[threads[thread]]["LABEL"].value_counts() for thread in threads.keys()}
    return pd.DataFrame(data=raw).T.fillna(0)


def contextual_resample(preprocessed: pd.DataFrame, subjects: pd.Series, size):
    """
    Resample in the context of the email dataset: threads will not be split
    """

    # CONTENT, LABEL

    threads = get_threads(subjects)
    labels = preprocessed["LABEL"]

    # list of labels and how many needed, with least occurring label coming first
    optimal_counts = (labels.value_counts() / labels.shape[0] * size).sort_values()

    candidates = []
    thread_count = 0

    labels_per_threads = get_labels_per_threads(threads, preprocessed)
    for label, amount in optimal_counts.iteritems():
        label_counts = labels_per_threads[label]

        # get all threads that would collect a label appropriately
        possible_threads = labels_per_threads[(label_counts > 0) & (label_counts <= amount)]

        while possible_threads.shape[0] == 0 or labels.iloc[candidates].value_counts().get(label, 0) < amount:
            print(possible_threads)

            chosen_thread = random.choice(possible_threads.index)
            chosen_emails = threads[chosen_thread]
            candidates.extend(chosen_emails)

            labels_per_threads = labels_per_threads.drop(chosen_thread)
            label_counts = labels_per_threads[label]

            # possible threads is now less, as allowed size has been shortened
            possible_threads = labels_per_threads[(label_counts > 0) & (label_counts <= amount)]

            thread_count += 1

    print(len(candidates), "in subset")
    print(thread_count, "threads")
    return candidates


if __name__ == '__main__':
    df = pd.read_csv("input.csv")
    pp = preprocess(df)
    cand = contextual_resample(pp, df["SUBJECT"], floor(df.shape[0] * 0.2))
    subset = pp.iloc[cand]  # ~20%
    altset = pp.drop(subset.index)  # ~80%

    print("total dist")
    print(pp["LABEL"].value_counts() / pp.shape[0])

    print("subset dist")
    print(subset["LABEL"].value_counts() / subset.shape[0])

    orig_rank_name = "ORIGINAL_RANK"
    folder = f"{int(time())}_division"
    os.mkdir(folder)

    subset.to_csv(os.path.join(folder, "test_set.csv"), index=True, index_label=orig_rank_name)
    altset.to_csv(os.path.join(folder, "training_set.csv"), index=True, index_label=orig_rank_name)
    pp.to_csv(os.path.join(folder, "both_preprocessed.csv"), index=True, index_label=orig_rank_name)
