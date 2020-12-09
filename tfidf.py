import pandas as pd
from nltk.corpus import stopwords
from pandas.core.dtypes.missing import notnull
from config import *
import io
import string
import csv
from math import log

STOP_WORDS = set(stopwords.words('english'))
PUNCTUATIONS = list(string.punctuation)
VOCAB = pd.read_csv(METADATA, delimiter="\t", header=None)
PUNCTUATIONS = list(string.punctuation)
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def read_data():
    data_train = pd.read_csv(TRAIN, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep="\t", header=None)
    data_test = pd.read_csv(TEST, quoting=csv.QUOTE_NONE, error_bad_lines=False, sep="\t", header=None)
    # print(data.shape, data)
    data = pd.concat([data_train, data_test], ignore_index=True)
    data.drop([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], axis=1, inplace=True)
    data.columns = [0, 1]
    # print(data)
    return data

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

def calculate_tfidf(sequences):
    global VOCAB
    max_length = 0
    for sequence in sequences:
        if max_length < len(sequence): max_length = len(sequence)
    out_tfidf = io.open('./data/tfidf.tsv', 'w', encoding='utf-8')
    n = len(VOCAB)
    df_dict= {}
    for row in range(n):
        word = VOCAB.iloc[row][0]
        # print(word)
        df = 0
        for sequence in sequences:
            df += 1
        df_dict[word] = df
    for sequence in sequences:
        sequence_tfidf_vector = []
        for token in sequence:
            f = 0
            for other_token in sequence:
                if token == other_token:
                    f += 1
            sequence_tfidf_vector.append((f / len(sequence)) * log(n / df_dict[token] + 1))
        for i in range(max_length - len(sequence)):
            sequence_tfidf_vector.append(0)
        # print(sequence_tfidf_vector)
        out_tfidf.write('\t'.join([str(x) for x in sequence_tfidf_vector]) + "\n")


if __name__ == "__main__":
    data = read_data()
    sequences = vectorize_sentences(data)
    calculate_tfidf(sequences)
