from config import Config
from buildVocabulary import getVocabulary
from model import taggingModel
if __name__ == "__main__":
    getVocabulary()
    config=Config()
    model=taggingModel(config)
    model.build()
    model.train(config.filename_train,config.filename_dev)

