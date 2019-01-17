# -*- coding:UTF-8 -*-



from config import Config
from buildVocabulary import getVocabulary
from model import taggingModel
if __name__ == "__main__":
    getVocabulary()
    config=Config()
    model=taggingModel(config)
    model.build()
    model.restore_session(config.dir_model_storepath)
    #sentence=raw_input('input your test sentences:\n')
    sentence="日报"
    model.predict(sentence)
