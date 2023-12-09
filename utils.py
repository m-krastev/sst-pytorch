import argparse
from math import prod
import os
import random
import re
from typing import NamedTuple
import zipfile
from nltk import Tree
from collections import Counter, OrderedDict
import numpy as np
import requests
from torch.utils.data import Dataset

import pytorch_lightning as pl


import torch
from tqdm import tqdm


def filereader(path):
    """this function reads in a textfile and fixes an issue with "\\\\\\" """
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))


class Example(NamedTuple):
    tokens: list
    tree: Tree
    label: int
    transitions: list


def examplereader(path, lower=False):
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)


def print_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += prod(p.shape)
        print(
            "{:24s} {:12s} requires_grad={}".format(
                name, str(list(p.shape)), p.requires_grad
            )
        )
    print("\nTotal number of parameters: {}".format(total))


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # cuda determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_file(url, savepath):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(savepath, "wb") as file, tqdm(
        desc=savepath,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return savepath


def load_embeddings(type, data_dir="data/"):
    """Load pretrained word embeddings from a file, returns path to embeddings file.
    Possible options: word2vec, glove"""
    match type:
        case "word2vec":
            savepath = f"{data_dir}/word2vec.txt"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            if os.path.exists(f"{data_dir}/word2vec.txt"):
                return f"{data_dir}/word2vec.txt"

            return download_file(
                "https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt",
                savepath,
            )

        case "glove":
            savepath = f"{data_dir}/glove.txt"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            if os.path.exists(savepath):
                return savepath
            return download_file(
                "https://gist.githubusercontent.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt",
                savepath,
            )
    raise TypeError("Unknown embedding type")


def load_dataset(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    if not os.path.exists(f"{savepath}/trees"):
        _ = download_file(
            "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip",
            savepath + "/trees.zip",
        )

        zipfile.ZipFile(_).extractall(savepath)

    return {
        "train": f"{savepath}/trees/train.txt",
        "dev": f"{savepath}/trees/dev.txt",
        "test": f"{savepath}/trees/test.txt",
    }


def initialize_vocabulary(embeddings_path):
    v = Vocabulary()
    v.add_token("<unk>")
    v.add_token("<pad>")

    vectors = []

    with open(embeddings_path, encoding="utf-8") as f:
        for line in f:
            word, embedding = line.strip().split(maxsplit=1)
            v.add_token(word)
            vectors.append(torch.tensor(list(map(float, embedding.split()))))
    # TODO: Justify using randn in report. Basically, zero-initialized vectors do not train as well.
    # We can also consider using zero-initialized vectors for the <unk> and <pad> tokens. (but then again it doesn't make sense if we train them)
    vectors.insert(0, torch.randn(len(vectors[0])))
    vectors.insert(1, torch.randn(len(vectors[0])))

    return v, torch.row_stack(vectors)


class TreeDataset(Dataset):
    def __init__(self, data: list[Example], vocab: Vocabulary):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = [self.vocab.w2i.get(t, 0) for t in self.data[idx].tokens]
        y = [self.data[idx].label]

        return torch.tensor(x).long(), torch.tensor(y).long()

    def __getitems__(self, indices):
        # x = [[self.vocab.w2i.get(t, 0) for t in self.data[idx].tokens] for idx in indices]
        # y = [self.data[idx].label for idx in indices]
        # return torch.tensor(x).long(), torch.tensor(y).long()
        # return [self.__getitem__(i) for i in indices]

        mb = [self.data[idx] for idx in indices]
        maxlen = max(len(ex.tokens) for ex in mb)

        # vocab returns 0 if the word is not there
        x = torch.tensor(
            [pad([self.vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
        ).long()
        y = torch.tensor([ex.label for ex in mb]).long()
        return x, y


def collate_batch(batch):
    return batch


class TreeDatasetPL(pl.LightningDataModule):
    def __init__(
        self,
        vocab: Vocabulary = None,
        batch_size: int = 32,
        num_workers: int = 4,
        data_dir: str = "data/",
        lower: bool = False,
    ):
        super().__init__()

        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lower = lower
        self.data_dir = data_dir

    def prepare_data(self):
        datasets = load_dataset(self.data_dir)

        if self.vocab is None:
            self.vocab = Vocabulary()
            for ex in examplereader(datasets["train"], lower=self.lower):
                for token in ex.tokens:
                    self.vocab.count_token(token)
            self.vocab.build()

        self.train_data = TreeDataset(
            list(examplereader(datasets["train"], lower=self.lower)), self.vocab
        )
        self.val_data = TreeDataset(
            list(examplereader(datasets["dev"], lower=self.lower)), self.vocab
        )
        self.test_data = TreeDataset(
            list(examplereader(datasets["test"], lower=self.lower)), self.vocab
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )


# naming convention goals
class TreeDatasetTree(Dataset):
    def __init__(self, data: list[Example], vocab: Vocabulary):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = [self.vocab.w2i.get(t, 0) for t in self.data[idx].tokens]
        y = [self.data[idx].label]

        return torch.tensor(x).long(), torch.tensor(y).long()

    def __getitems__(self, indices):
        """
        Returns sentences reversed (last word first)
        Returns transitions together with the sentences.
        """
        mb = [self.data[idx] for idx in indices]

        maxlen = max([len(ex.tokens) for ex in mb])

        # vocab returns 0 if the word is not there
        # NOTE: reversed sequence!
        x = torch.tensor(
            [
                pad([self.vocab.w2i.get(t, self.vocab.w2i["<unk>"]) for t in ex.tokens], maxlen)[::-1]
                for ex in mb
            ],
            dtype=torch.int64,
        )
        y = torch.tensor([ex.label for ex in mb], dtype=torch.int64)

        maxlen_t = max([len(ex.transitions) for ex in mb])
        transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
        transitions = np.array(transitions)
        transitions = transitions.T  # time-major

        return (x, transitions), y


class TreeDatasetTreePL(pl.LightningDataModule):
    def __init__(
        self,
        vocab: Vocabulary = None,
        batch_size: int = 32,
        num_workers: int = 4,
        data_dir: str = "data/",
        lower: bool = False,
        subtrees: bool = False,
    ):
        super().__init__()

        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lower = lower
        self.data_dir = data_dir
        self.subtrees = subtrees

    def prepare_data(self):
        datasets = load_dataset(self.data_dir)

        if self.vocab is None:
            self.vocab = Vocabulary()
            for ex in examplereader(datasets["train"], lower=self.lower):
                for token in ex.tokens:
                    self.vocab.count_token(token)
            self.vocab.build()

        if self.subtrees:
            # create subtrees
            train_data = list(examplereader(datasets["train"], lower=self.lower))
            subtrees = []
            for ex in train_data:
                for subtree in ex.tree.subtrees():
                        print(subtree.__str__())
                        subtrees.append(
                            Example(
                                tokens=subtree.leaves(),
                                tree=subtree,
                                label=subtree.label(),
                                transitions=transitions_from_treestring(
                                    subtree.__str__()
                                ),
                            )
                        )
            self.train_data = TreeDatasetTree(subtrees, self.vocab)
            
            val_data = list(examplereader(datasets["dev"], lower=self.lower))
            subtrees = []
            for ex in val_data:
                for subtree in ex.tree.subtrees():
                    subtrees.append(
                        Example(
                            tokens=subtree.leaves(),
                            tree=subtree,
                            label=subtree.label(),
                            transitions=transitions_from_treestring(
                                subtree.__str__()
                            ),
                        )
                    )
            self.val_data = TreeDatasetTree(subtrees, self.vocab)
            
            test_data = list(examplereader(datasets["test"], lower=self.lower))
            subtrees = []
            for ex in test_data:
                for subtree in ex.tree.subtrees():
                    subtrees.append(
                        Example(
                            tokens=subtree.leaves(),
                            tree=subtree,
                            label=subtree.label(),
                            transitions=transitions_from_treestring(
                                subtree.__str__()
                            ),
                        )
                    )
            self.test_data = TreeDatasetTree(subtrees, self.vocab)
        else:
            self.train_data = TreeDatasetTree(
                list(examplereader(datasets["train"], lower=self.lower)), self.vocab
            )
            self.val_data = TreeDatasetTree(
                list(examplereader(datasets["dev"], lower=self.lower)), self.vocab
            )
            self.test_data = TreeDatasetTree(
                list(examplereader(datasets["test"], lower=self.lower)), self.vocab
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )


def parser():
    """
    Parse command line arguments for the NLP model.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    # Snellius seems to sometimes not detect CUDA
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="save/",
        help="location of the model file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="number of workers for the dataloader",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots/",
        help="location of the plots file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="location of the data directory, should contain train.txt, dev.txt, test.txt, as well as the word2vec embeddings",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="location of the results file",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="whether to only evaluate"
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="name of the model file"
    )
    parser.add_argument("--hidden_dim", type=int, default=150, help="hidden dimension")
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[100, 100],
        help="hidden dimensions",
    )
    parser.add_argument(
        "--embeddings_type",
        type=str,
        default=None,
        choices=["word2vec", "glove"],
        help="use pretrained embeddings",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=10000, help="number of iterations"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=300, help="dimension of embeddings"
    )
    parser.add_argument("--print_every", type=int, default=1000, help="print every")
    parser.add_argument("--eval_every", type=int, default=1000, help="evaluate every")
    parser.add_argument(
        "--classes",
        default=["very negative", "negative", "neutral", "positive", "very positive"],
        nargs="+",
        help="classes",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--lower", action="store_false", help="do not lowercase the data"
    )
    parser.add_argument(
        "--train_embeddings",
        action="store_true",
        help="whether to train embeddings along with model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="name of the model checkpoint file",
    )

    parser.add_argument(
        "--subtrees",
        action="store_true",
        help="whether to use subtrees",
    )

    return parser.parse_args()
