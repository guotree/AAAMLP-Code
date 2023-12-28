import os
import transformers

# fetch home directory
# in our docker container, it is
# /home/abhishek
HOME_DIR = os.path.expanduser("~")
# this is the maximum number of tokens in the sentence
MAX_LEN = 512
# batch sizes is low because model is huge!
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# let's train for a maximum of 10 epochs
EPOCHS = 10
# define path to BERT model files
# Now we assume that all the data is stored inside
# /home/abhishek/data
BERT_PATH = os.path.join(HOME_DIR, "data", "bert_base_uncased")
# this is where you want to save the model
MODEL_PATH = os.path.join(HOME_DIR, "data", "model.bin")
# training file
TRAINING_FILE = os.path.join(HOME_DIR, "data", "imdb.csv")
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
