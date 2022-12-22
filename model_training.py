import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
import nltk
from data_set_loader import get_data_set_loader
from pipeline_models import EncoderCNN, DecoderRNN
import math
import numpy as np
import os
import time
#All the parameters required to train the cnn-rnn model architecture
batch_size = 512
vocabulary_threshold = 6
#False if we had the file created once while training. Pass it as True if vocabulary file has to be generated from scratch.
vocabulary_from_file = False
embedding_size = 512
hidden_units = 512

#Log file consisting of the loss and perplexity for each batch while training
training_log_file = 'training_log.txt'

nltk.download('punkt')

#Images are transformed before being sent to the CNN model in the following way.
transform_train = transforms.Compose([ 
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

#Load the dataset in train mode.
data_set_loader = get_data_set_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocabulary_threshold=vocabulary_threshold,
                         vocab_from_file=vocabulary_from_file)

#Size of the vocabulary 
vocabulary_size = len(data_set_loader.dataset.vocabulary)

#Initialize encoder and decoder models using the above set parameters.
encoder_cnn = EncoderCNN(embedding_size)
decoder_rnn = DecoderRNN(embedding_size, hidden_units, vocabulary_size,batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_cnn.to(device)
decoder_rnn.to(device)

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
parameters = list(decoder_rnn.parameters()) + list(encoder_cnn.embed.parameters())

#We are using adams optimizer and learning rate of 0.001 after hyper parameter tuning
optimizer = torch.optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
total_noof_steps = math.ceil(len(data_set_loader.dataset.caption_lengths) / data_set_loader.batch_sampler.batch_size)

file_ptr = open(training_log_file, 'w')
num_of_epochs = 10 
old_time = time.time()

for epoch_number in range(1, num_of_epochs+1):
    for step in range(1, total_noof_steps+1): 
        if time.time() - old_time > 60:
            old_time = time.time()
        indices = data_set_loader.dataset.get_train_indices()
        
        #Taking a sample of the data
        random_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_set_loader.batch_sampler.sampler = random_sampler
        
        #Getting the image and the captions for the image
        images, image_captions = next(iter(data_set_loader))
        images = images.to(device)
        image_captions = image_captions.to(device)
        
        decoder_rnn.zero_grad()
        encoder_cnn.zero_grad()
        
        #Get features of the image from the encoder CNN model
        image_features = encoder_cnn(images)

        #Get word indices as output from the decoder RNN model
        outputs = decoder_rnn(image_features, image_captions)
        
        #Calculate the loss and propogate it through the RNN network.
        loss = criterion(outputs.contiguous().view(-1, vocabulary_size), image_captions.view(-1))
        loss.backward()
        
        optimizer.step()
            
        stats = 'Epoch Number: [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch_number, num_of_epochs, step, total_noof_steps, loss.item(), np.exp(loss.item()))

        #Write the logs to the log file.         
        file_ptr.write(stats + '\n')
        file_ptr.flush()
        
        if step % 200 == 0:
            print('\r' + stats)

    #Save the encoder and decoder checkpoints in /models folder.        
    torch.save(decoder_rnn.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch_number))
    torch.save(encoder_cnn.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch_number))

file_ptr.close()