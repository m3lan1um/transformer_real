"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.data import Field, BucketIterator, TabularDataset
# from torchtlang.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, lang, tokenize_en, tokenize_fr, init_token, eos_token):
        self.lang = lang
        self.tokenize_en = tokenize_en
        self.tokenize_fr = tokenize_fr
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.lang == ('fr', 'en'):
            self.source = Field(tokenize=self.tokenize_fr, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True, fix_length=256)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True, fix_length=256)

        elif self.lang == ('en', 'fr'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True, fix_length=256)
            self.target = Field(tokenize=self.tokenize_fr, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True, fix_length=256)

        print('make dataset start') 
        train_data, valid_data, test_data = TabularDataset.splits(
            path='/home/cvmlserver4/junhee/transformer/datasets', train='train.csv', validation='valid.csv', test='test.csv', format='csv',
            fields=[(self.lang[0], self.source), (self.lang[1], self.target)]
        )
        print('make dataset finish')

        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,    
                                                                              sort_key = lambda x: len(x.en), 
                                                                              sort_within_batch=True,
                                                                              device=device)
        print('dataset initializing done')
        print(train_iterator)
        return train_iterator, valid_iterator, test_iterator
