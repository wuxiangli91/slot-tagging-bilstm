import os
from dataUtils import load_vocab


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger

        # load if requested (default)
        if load:
            self.load()



    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.dWords = load_vocab(self.filename_words)
        self.dTags  = load_vocab(self.filename_tags,isTags=True)
        self.dIdToTags = {value: key for key, value in self.dTags.items()}
        self.nwords     = len(self.dWords)
        self.ntags      = len(self.dTags)

        # 2. get pre-trained embeddings
        self.embeddings=None


    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "20epochmodels/"
    dir_model_storepath = dir_output + "20epochmodels/"
    model_prefix = 'model-ckpt'
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300

    # glove files
    use_pretrained = False

    # dataset
    filename_dev = "/home/wxl/PycharmProjects/slotTag/data/mydata/dev.txt"
    filename_test = "/home/wxl/PycharmProjects/slotTag/data/mydata/test.txt"
    filename_train = "/home/wxl/PycharmProjects/slotTag/data/mydata/train.txt"
    filename_rawData = "/home/wxl/PycharmProjects/slotTag/data/rawTrain.txt"
    #filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = True
    nepochs          = 20
    dropout          = 0.5
    batch_size       = 200
    dev_batch_size   =1000
    test_batch_size  =1
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

