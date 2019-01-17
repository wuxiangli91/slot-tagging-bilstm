
def load_vocab(dataFile,isTags=False):
    """Loads vocab from a file
        Args:
            filename: (string) the format of the file must be one word per line.
        Returns:
            d: dict[word] = index
    """
    d = dict()
    with open(dataFile) as f:
        for idx, word in enumerate(f):
            word = word.replace('\r', '').replace('\n', '')
            d[word] = idx

    #add 'UNK' and 'PAD' to dict
    if not isTags:
        d['UNK']=len(d)
        d['PAD']=len(d)

    return d

def getBatches(dataFile,minibatch_size,dWords,dTags):
    x_batch,y_batch=[],[]
    with open(dataFile) as fr:
        cnt=0
        x,y=[],[]
        for line in fr:
            line=line.replace('\r','').replace('\t','').split(' ')[:-1]
            if cnt%2==0:
                for item in line:
                    x.append(dWords[item] if item in dWords.keys() else dWords['UNK'])
                x_batch.append(x)

            else:
                for item in line:
                    y.append(dTags[item])
                y_batch.append(y)
                y=[]
                x=[]

            cnt+=1

            if len(x_batch)==minibatch_size and len(y_batch)==minibatch_size:
                yield  x_batch,y_batch
                x_batch,y_batch=[],[]

    if len(x_batch)!=0:
        yield x_batch,y_batch





def pad_sequences(sequences, pad_tok):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
    a list of list where each sublist has same length
    """
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length