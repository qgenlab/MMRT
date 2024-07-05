from sklearn.utils import shuffle as sh
import torch
import random
import os


class Logger:
    def __init__(self, path, name, resume_log = False, delim = ',', columns = ['col1']):
        self.path = path
        self.name = name
        self.delim = delim
        self.column_width = len(columns)
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.file = open(path + name + '.csv', 'a' if resume_log else 'w')
        
        if not(resume_log):
            self.file.write(delim.join(columns) + '\n')
            self.file.flush()
        
    def __del__(self):
        self.file.close()
        
    def write(self, line):
        assert len(line) == self.column_width, f"Length of line should equal number of columns ({self.column_width})"
        self.file.write(self.delim.join(map(str, line)) + '\n')
        self.file.flush()

        
class Loader:
    """
    We need to make our own dataloader to handle cases in which we are
    training or testing on multiple orders of mutations (single, double, etc).
    
    This expects an iterable of datasets (where a "dataset" is a tuple of
    mutation strings, ESM vectors, and the activity scores, as produced by
    generate_vectors.py), and has the option to shuffle, and select an integer
    batch size.
    
    Essentially, this dataloader is just a list of PyTorch's default DataLoaders,
    one for each dataset. At each iteration, a dataloader is chosen at random,
    and a batch is returned from it.
    
    Once a dataloader has been "used up," it is removed from the list, until
    the list is empty.
    """
    def __init__(self, data, shuffle=True, batch_size=None):
        if batch_size is None:
            self.batch_size = max([len(d[0]) for d in data])
        else:
            self.batch_size = batch_size
        
        self.dataloaders = [
            torch.utils.data.DataLoader(
                list(zip(*d)),
                shuffle=shuffle,
                batch_size=self.batch_size
            ) for d in data
        ]

    def __iter__(self):
        self.iter = [iter(x) for x in self.dataloaders]
        return self
    
    def __next__(self):
        while True:
            if not(self.iter):
                raise StopIteration
            try:
                current = random.choice(self.iter)
                return next(current)
            except StopIteration:
                self.iter.remove(current)
        
    def __len__(self):
        return sum([len(l) for l in self.dataloaders])
    
    
def train_test_split(data, train_ratio=0.9, shuffle=False):
    train=[]
    test=[]
    for d in data:
        if shuffle:
            s, v, a = sh(*d)
        else:
            s, v, a = d
        split = int(ratio * len(a))
        train_s, train_v, train_a = s[:split], v[:split], a[:split]
        test_s, test_v, test_a = s[split:], v[split:], a[split:]
        train.append((train_s, train_v, train_a))
        test.append((test_s, test_v, test_a))
    
    return train, test