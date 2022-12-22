import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):

    def __init__(self,
        vocabulary_threshold,
        vocabulary_file='vocab.pkl',
        starting_word="<start>",
        ending_word="<end>",
        unknown_word="<unk>",
        annotations_file='captions_train2014.json',
        vocab_from_file=False):
        self.vocabulary_threshold = vocabulary_threshold
        self.vocabulary_file = vocabulary_file
        self.starting_word = starting_word
        self.ending_word = ending_word
        self.unknown_word = unknown_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocabulary_file) & self.vocab_from_file:
            with open(self.vocabulary_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary completely loaded..')
        else:
            self.build_vocab()
            with open(self.vocabulary_file, 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.starting_word)
        self.add_word(self.ending_word)
        self.add_word(self.unknown_word)
        self.add_captions()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocabulary_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)