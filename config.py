DATASET_ROOT = "./datasets/liar_dataset/"
TRAIN = DATASET_ROOT + "train.tsv"
TEST = DATASET_ROOT + "test.tsv"
WRITE_FILE = "./data/credibility_lookup.pickle"
LABEL_MAPPING = {
    "true": 5,
    "mostly-true": 4,
    "half-true": 3,
    "barely-true": 2,
    "false": 1,
    "pants-fire": 0
}

EMBEDDING_DIM = 128
WINDOW_SIZE = 2
NUM_NS = 4
SEED = 45
