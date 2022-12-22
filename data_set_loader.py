import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import json

def get_data_set_loader(transform,
               mode='train',
               batch_size=1,
               vocabulary_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               base_path='/Users/vamshi/Desktop/image-captioning-github/image-captioning'):
    
    """ 
        Data loader to get training/testing data based on mode. 
        Please change the paths as per your local paths.

        Arguments:
            mode: 'train'/'test'.
            batch_size: Size of the batch. Its value has to be 1 in case of test
            vocabulary_threshold: Minimum number of words threshold.
            vocab_file: Vocabulary file. 
            start_word: Special token which will indicate start of the sentence.
            end_word: Special token which will indicate end of the sentence.
            unk_word: Special token which will indicate unknown words.
            vocab_from_file: Vocabulary file will be created from scratch if this flag is True. Else existing vocabulary file will be used.
            num_workers: Degree of parallelization
            base_path: Location of the base folder which consists of these files
    """

    if vocab_from_file==False: 
        if(mode!='train'):
            print("To generate vocab from captions file, must be in training mode.")
    if mode == 'train':
        if vocab_from_file==True:
            assert os.path.exists(vocab_file), "vocab_file is not present.  Send vocab_from_file parameter as False to create vocab_file."
        img_folder = os.path.join(base_path, 'train2014/')
        annotations_file = os.path.join(base_path, 'captions_train2014.json')
    if mode == 'test':
        if(batch_size!=1):
            "Batch_size has to be 1 if mode is test"
        assert os.path.exists(vocab_file), "To test the model, vocab.pkl has to be generated first from training data."
        if(vocab_from_file!=True):
            "Change vocab_from_file to True."
        img_folder = os.path.join(base_path, 'test2014/')
        annotations_file = os.path.join(base_path, 'image_info_test2014.json')

    dataset = MS_Coco_DataSet(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocabulary_threshold=vocabulary_threshold,
                          vocabulary_file=vocab_file,
                          starting_word=start_word,
                          ending_word=end_word,
                          unknown_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        indices = dataset.get_train_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                    drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

#MSCOCO Dataset class
class MS_Coco_DataSet(data.Dataset):
    #Initialize the coco data set class
    def __init__(self, transform, mode, batch_size, vocabulary_threshold, vocabulary_file, starting_word, 
        ending_word, unknown_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocabulary = Vocabulary(vocabulary_threshold, vocabulary_file, starting_word,
            ending_word, unknown_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            tokens=[]
            for ind in tqdm(np.arange(len(self.ids))):
                caption_str=str(self.coco.anns[self.ids[ind]]['caption']).lower()
                tokenized_caption=nltk.tokenize.word_tokenize(caption_str)
                tokens.append(tokenized_caption)
            self.caption_lengths = [len(token) for token in tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            paths=[]
            for item in test_info['images']:
                paths.append(item)
            self.paths=paths
    
    #Get each image item from the dataset.
    def __getitem__(self, index):
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            img_path = self.coco.loadImgs(img_id)[0]['file_name']

            image = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')
            image = self.transform(image)

            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocabulary(self.vocabulary.starting_word))
            caption.extend([self.vocabulary(token) for token in tokens])
            caption.append(self.vocabulary(self.vocabulary.ending_word))
            caption = torch.Tensor(caption).long()
            return image, caption

        else:
            path = self.paths[index]
            PIL_image = Image.open(os.path.join(self.img_folder, path["file_name"])).convert('RGB')
            original_image = np.array(PIL_image)
            transformed_image = self.transform(PIL_image)
            return original_image, transformed_image

    #Get the training indices for the captions by randomizing them.
    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    #Return the length of ids in train mode and
    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)