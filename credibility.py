import pandas as pd
import pickle
from utils.utils import return_label_weights
from config import *

CREDIBILITY = {}


def read_data():
    data = pd.read_csv(TRAIN, delimiter="\t", header=None)
    data.append(pd.read_csv(TEST, delimiter="\t", header=None))
    data.dropna(axis=0, inplace=True)
    data.drop([0, 2, 5, 6, 8, 9, 10, 11, 12, 13], axis=1, inplace=True)
    data.columns = [0, 1, 2, 3]
    print(data.head())
    return data


def generate_credibility(data):
    for i in range(len(data)):
        if not data.iloc[i][2] in CREDIBILITY:
            CREDIBILITY[data.iloc[i][2]] = {}
        for subject in data.iloc[i][1].split(","):
            if not subject in CREDIBILITY[data.iloc[i][2]]:
                CREDIBILITY[data.iloc[i][2]][subject] = {
                    "total": 0,
                    "value": 0
                }
            CREDIBILITY[data.iloc[i][2]][subject]["total"] += 1
            CREDIBILITY[data.iloc[i]
                        [2]][subject]["value"] += return_label_weights(
                            LABEL_MAPPING[data.iloc[i][0]])

    for person in CREDIBILITY:
        cumulative = 0
        for subject in CREDIBILITY[person]:
            CREDIBILITY[person][subject]["value"] = round(
                CREDIBILITY[person][subject]["value"] /
                CREDIBILITY[person][subject]["total"], 4)
            cumulative += CREDIBILITY[person][subject]["value"]
        CREDIBILITY[person]["cumulative"] = round(
            cumulative / len(CREDIBILITY[person].keys()), 2)


def dump_credibility_object():
    with open(WRITE_FILE, "wb") as file:
        pickle.dump(CREDIBILITY, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data = read_data()
    generate_credibility(data)
    dump_credibility_object()
# print(list(CREDIBILITY.items())[0:2])