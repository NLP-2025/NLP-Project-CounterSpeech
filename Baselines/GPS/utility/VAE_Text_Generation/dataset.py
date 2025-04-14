import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from gensim.models import KeyedVectors

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

my_punc = "!\"#$%&\()*+?_/:;[]{}|~,`"
table = dict((ord(char), ' ') for char in my_punc)

def clean_str(string):
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"\'m ", " ", string)
    string = re.sub(r"\'ve ", " ", string)
    string = re.sub(r"n\'t ", " not ", string)
    string = re.sub(r"\'re ", " ", string)
    string = re.sub(r"\'d ", " ", string)
    string = re.sub(r"\'ll ", " ", string)
    string = re.sub("-", " ", string)
    string = re.sub(r"@", " ", string)
    string = re.sub('\'', '', string)
    string = string.translate(table)
    string = string.replace("..", "").strip()
    return string

def tokenize(text):
    return [tok for tok in clean_str(text).strip().split() if tok]

def build_vocab(dataset, glove_model, max_vocab):
    counter = Counter()
    for line in dataset:
        counter.update(tokenize(line))
    most_common = [tok for tok, _ in counter.most_common(max_vocab - len(SPECIAL_TOKENS))]

    itos = SPECIAL_TOKENS + most_common
    stoi = {tok: i for i, tok in enumerate(itos)}

    emb_dim = glove_model.vector_size
    embeddings = torch.randn(len(itos), emb_dim)
    for i, word in enumerate(itos):
        if word in glove_model:
            embeddings[i] = torch.tensor(glove_model[word])
        elif word in SPECIAL_TOKENS:
            embeddings[i] = torch.zeros(emb_dim)

    vocab = {
        'stoi': stoi,
        'itos': itos,
        'embedding_matrix': embeddings
    }
    return vocab

class MyDataset(Dataset):
    def __init__(self, filepath, vocab):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f.readlines()]
        self.stoi = vocab['stoi']

    def encode(self, tokens):
        ids = [self.stoi.get(SOS_TOKEN)]
        ids += [self.stoi.get(tok, self.stoi.get(UNK_TOKEN)) for tok in tokens]
        ids += [self.stoi.get(EOS_TOKEN)]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        return self.encode(tokens)

    def __len__(self):
        return len(self.texts)

def collate_batch(batch, pad_idx):
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)

def get_iterators(opt, path, fname, glove_path):
    full_path = f"{path}/{fname}"

    with open(full_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Load GloVe
    glove = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    # Vocab build
    vocab = build_vocab(raw_lines, glove, opt.n_vocab)

    # Dataset
    dataset = MyDataset(full_path, vocab)

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab['stoi'][PAD_TOKEN])
    )

    # Match original function return
    return dataloader, dataloader, vocab
