import os
from config import Config
def getVocabulary():

    config = Config(load=False)
    rawDataPath = config.filename_rawData
    
    with open(rawDataPath) as fr:
        cnt=0
        vocabWords = set()
        vocabTags = set()
        for line in fr:
            line = line.strip().split(' ')
            if cnt%2==0:
                vocabWords.update(line)
            else:
                vocabTags.update(line)
            cnt+=1
        with open(config.filename_words,'w') as fw:
            for x in vocabWords:
                fw.write("{}\n".format(x))
        with open(config.filename_tags,'w') as fw:
            for x in vocabTags:
                fw.write("{}\n".format(x))
