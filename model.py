import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from nltk.corpus import stopwords
from utils.utils import *
from config import *
import io

STOP_WORDS = set(stopwords.words('english'))
VOCAB = {}
INVERSE_VOCAB = {}
VOCAB_SIZE = 0
TRAIN_DATA = []
AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_data():
    data = pd.read_csv(TRAIN, delimiter="\t", header=None)
    # data.append(pd.read_csv(TEST, delimiter="\t", header=None))
    data.dropna(axis=0, inplace=True)
    data.drop([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], axis=1, inplace=True)
    data.columns = [0, 1]
    return data


def combine_labels(data):
    for i in range(len(data)):
        if data.iloc[i][0] in ["pants-fire", "false", "barely-true"]:
            data.iloc[i][0] = 0
        else:
            data.iloc[i][0] = 1
    return data


def generate_vocabulary(data):
    global VOCAB, INVERSE_VOCAB, VOCAB_SIZE
    index = 1
    for i in range(len(data)):
        for token in data.iloc[i][1].lower().split():
            if token not in STOP_WORDS:
                if token not in VOCAB:
                    VOCAB[token] = index
                    INVERSE_VOCAB[index] = token
                    index += 1
    VOCAB['<pad>'] = 0
    INVERSE_VOCAB[0] = '<pad>'
    VOCAB_SIZE = index


def vectorize_sentences(data):
    sequences = []
    max_length = 0
    for i in range(len(data)):
        sentence = []
        for token in data.iloc[i][1].lower().split():
            if token not in STOP_WORDS:
                sentence.append(VOCAB[token])
        if max_length < len(sentence): max_length = len(sentence)
        sequences.append(sentence)

    for i in range(len(sequences)):
        if len(sequences[i]) < max_length:
            for j in range(max_length - len(sequences[i]) + 1):
                sequences[i].append(0)

    return sequences


def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    print(vocab_size)
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        vocab_size)
    print("sampling_table : ", sampling_table)
    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=SEED,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates],
                                0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels


def generate_training_examples(sequences):
    global VOCAB, INVERSE_VOCAB, VOCAB_SIZE, WINDOW_SIZE, NUM_NS, SEED, AUTOTUNE
    targets, contexts, labels = generate_training_data(sequences=sequences,
                                                       window_size=WINDOW_SIZE,
                                                       num_ns=NUM_NS,
                                                       vocab_size=VOCAB_SIZE,
                                                       seed=SEED)
    print(len(targets), len(contexts), len(labels))
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,
                                                 drop_remainder=True)
    print(dataset)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print(dataset)
    return dataset


class Word2Vec(Model):
    global NUM_NS

    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name="w2v_embedding",
        )
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=NUM_NS + 1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


def train_word2vec_model(dataset):
    embedding_dim = EMBEDDING_DIM
    vocab_size = VOCAB_SIZE
    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
    return word2vec


def save_word_vectors(word2vec):
    global VOCAB
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = VOCAB
    out_v = io.open('./data/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('./data/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0: continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()


if __name__ == "__main__":
    data = read_data()
    data = combine_labels(data)
    generate_vocabulary(data)
    sequences = vectorize_sentences(data)
    dataset = generate_training_examples(sequences)
    word2vec = train_word2vec_model(dataset)
    save_word_vectors(word2vec)
    # print(data.head(), sequences[:2],
    #       list(VOCAB.items())[:4],
    #       list(INVERSE_VOCAB.items())[:4])
