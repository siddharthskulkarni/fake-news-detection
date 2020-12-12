from model import generate_training_data
from os import read
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from config import *
import string
from nltk.corpus import stopwords
import csv
import io
import json

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATIONS = list(string.punctuation)
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
VOCAB = {}
INVERSE_VOCAB = {}
VOCAB_SIZE = 0
TFIDF_VECTORS_DF = pd.read_csv(TFIDF_VECTORS, delimiter="\t", header=None)
WORD_VECTORS_DF = pd.read_csv(WORD_VECTORS, delimiter="\t", header=None)
LABELS_DF = pd.read_csv(LABELS, delimiter="\t", header=None)

def read_data():
    data_train = pd.read_csv(TRAIN, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep="\t", header=None)
    # data = data_train
    data_test = pd.read_csv(TEST, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep="\t", header=None)
    # print(data.shape, data)
    data = pd.concat([data_train, data_test], ignore_index=True)
    data.drop([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], axis=1, inplace=True)
    data.columns = [0, 1]
    # print(data)
    return data

def generate_vocabulary(data):
    global VOCAB, INVERSE_VOCAB, VOCAB_SIZE, PUNCTUATIONS
    index = 1
    for i in range(len(data)):
        for token in data.iloc[i][1].lower().strip().split():
            for j in range(len(PUNCTUATIONS)):
                token = token.replace(PUNCTUATIONS[j], "")
            for j in range(len(DIGITS)):
                token = token.replace(DIGITS[j], "")
            if token not in STOP_WORDS and token != "":
                if token not in VOCAB:
                    if token == "null": token = "nul"
                    VOCAB[token] = index
                    INVERSE_VOCAB[index] = token
                    index += 1
    VOCAB['<pad>'] = 0
    INVERSE_VOCAB[0] = '<pad>'
    VOCAB_SIZE = index

def vectorize_sentences(data):
    global PUNCTUATIONS, DIGITS
    sequences = []
    for i in range(len(data)):
        sentence = []
        for token in data.iloc[i][1].lower().split():
            for j in range(len(PUNCTUATIONS)):
                token = token.replace(PUNCTUATIONS[j], "")
            for j in range(len(DIGITS)):
                token = token.replace(DIGITS[j], "")
            if token not in STOP_WORDS and token != "":
                if token == "null": token = "nul"
                sentence.append(token)
        sequences.append(sentence)
    # print("seq = ",sequences)
    return sequences

def generate_input_vectors(sequences):
    out_input_vectors = io.open('./data/input_vectors.tsv', 'w', encoding='utf-8')
    out_input_vectors_string = ""
    out_input_vectors_words = io.open('./data/input_vectors_words.tsv', 'w', encoding='utf-8')
    out_input_vectors_words_string = ""
    global TFIDF_VECTORS_DF, WORD_VECTORS_DF, INPUT_VECTORS
    input_vectors = []
    for i in range(len(TFIDF_VECTORS_DF)):
        # print(sequences[i])
        sequence_vector = [0 for j in range(EMBEDDING_DIM)]
        for j in range(EMBEDDING_DIM):
            cum_tfidf = 0
            for k in range(len(sequences[i])):
                sequence_vector[j] += WORD_VECTORS_DF.iloc[VOCAB[sequences[i][k]] - 1][j]
                # cum_tfidf += TFIDF_VECTORS_DF.iloc[i][k]
                # try: print(WORD_VECTORS_DF.iloc[VOCAB[sequences[i][k]]][j])
                # except Exception as error: print(error, sequences[i][k])
            # sequence_vector[j] /= cum_tfidf
        print("Tuple ", i)
        # out_input_vectors.write("\t".join(str(x) for x in sequence_vector) + "\n")
        # out_input_vectors_words.write("\t".join(x for x in sequences[i]) + "\n")
        out_input_vectors_string += "\t".join(str(x) for x in sequence_vector) + "\n"
        out_input_vectors_words_string += "\t".join(x for x in sequences[i]) + "\n"
        input_vectors.append(sequence_vector)
    # print(input_vectors[:2])
    out_input_vectors.write(out_input_vectors_string)
    out_input_vectors_words.write(out_input_vectors_words_string)
    return input_vectors

def test_models(input_vectors, models):
    global LABELS_DF
    output = {}
    output["combined"] = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "acc": 0
    }
    for key in models:
        output[key] = {
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "acc": 0
            }
    # tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(N_TRAIN + 1, N_TOTAL):
        train_tuple = input_vectors.iloc[i]
        labelled_val = LABELS_DF.iloc[i][0]
        tupple_class = 0
        for key in models:
            predicted_val = models[key].predict([train_tuple])
            tupple_class += predicted_val
            # print(LogisticRegressionModel.predict([train_tuple]), LABELS_DF.iloc[i][0])
            if predicted_val == labelled_val:
                if predicted_val == 0:
                    output[key]["tp"] += 1
                else:
                    output[key]["tn"] += 1
            else:
                if predicted_val == 1:
                    output[key]["fn"] += 1
                else:
                    output[key]["fp"] += 1
        if tupple_class < 2:
            if labelled_val == 0:
                output["combined"]["tp"] += 1
            else:
                output["combined"]["fp"] += 1
        else:
            if labelled_val == 1:
                output["combined"]["tn"] += 1
            else:
                output["combined"]["fn"] += 1

    # output[key]["acc"] = (output[key]["tp"] + output[key]["tn"]) / (output[key]["tp"] + output[key]["tn"] + output[key]["fp"] + output[key]["fn"])
    # print("Model = ", key)
    # print("TP    = ", output[key]["tp"])
    # print("TN    = ", output[key]["tn"])
    # print("FP    = ", output[key]["fp"])
    # print("FN    = ", output[key]["fn"])
    # print("ACC   = ", output[key]["acc"])
    
    for key in output:
        output[key]["acc"] = (output[key]["tp"] + output[key]["tn"]) / (output[key]["tp"] + output[key]["tn"] + output[key]["fp"] + output[key]["fn"])
    print(output)
    with open("./data/output_word2vec.json", "w+") as f:
        json.dump(output, f)
    # json.dump("./data/output_word2vec.tsv")



def logistic_regression(X, y):
    # X, y = load_iris(return_X_y=True)
    # print(X, y)
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf

def k_nearest_neighbours(X, y):
    # X = [[0], [1], [2], [3]]
    # y = [0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, y)
    return neigh
    # neigh.fit(X, y)
    # print(neigh.predict([[1.1]]))
    # print(neigh.predict_proba([[0.9]]))

def support_vector_machine(X, y):
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

if __name__ == "__main__":
    # logistic_regression()
    # print("Reading Data...")
    # data = read_data()
    # print("Generating Vocabulary...")
    # generate_vocabulary(data)
    # print("Vectorizing sentences...")
    # sequences = vectorize_sentences(data)
    # print("Generating Training Data...")
    # input_vectors = generate_input_vectors(sequences)
    # # print(len(input_vectors[:N_TRAIN]), len(LABELS_DF[:N_TRAIN]))
    # # print(input_vectors[:N_TRAIN], LABELS_DF[:N_TRAIN])
    input_vectors = pd.read_csv(INPUT_VECTORS, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep="\t", header=None)
    print("Training Model...")
    LogisticRegressionModel = logistic_regression(input_vectors[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    KNeighbours = k_nearest_neighbours(input_vectors[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    SupportVectorMachine = support_vector_machine(input_vectors[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    # print("Testing...")
    # test_models(input_vectors, {"logistic_regression": LogisticRegressionModel, "k_nearest_neighbours": KNeighbours, "support_vector_machine": SupportVectorMachine})
    # print("Testing...2")
    # LogisticRegressionModel = logistic_regression(TFIDF_VECTORS_DF[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    # KNeighbours = k_nearest_neighbours(TFIDF_VECTORS_DF[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    # SupportVectorMachine = support_vector_machine(TFIDF_VECTORS_DF[:N_TRAIN], LABELS_DF[:N_TRAIN][0])
    print("Testing...")
    test_models(input_vectors, {"logistic_regression": LogisticRegressionModel, "k_nearest_neighbours": KNeighbours, "support_vector_machine": SupportVectorMachine})