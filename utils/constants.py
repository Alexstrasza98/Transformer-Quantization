import os

BATCH_SIZE = 32

BASELINE_MODEL_NUMBER_OF_LAYERS = 1
BASELINE_MODEL_DIMENSION = 512
BASELINE_MODEL_NUMBER_OF_HEADS = 8
BASELINE_MODEL_DROPOUT_PROB = 0.1
BASELINE_MODEL_LABEL_SMOOTHING_VALUE = 0.1


CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "checkpoints")
BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "binaries")
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "data")
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)


BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"