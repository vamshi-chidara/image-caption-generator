import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    #Initialize CNN model with given embedding size.
    #As an encoder, we will be using a pretrained Resnet50 model available in the pytorch library.
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for parameters in resnet.parameters():
            parameters.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embedding_size)

    #forward pass
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    #Initialize RNN model with given embedding size, number of hidden units, size of the vocabulary and batch size.
    def __init__(self, embedding_size, hidden_size, vocabulary_size, batch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, \
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0,
                            bidirectional=False)
        self.batch_size=batch_size
        self.linear = nn.Linear(hidden_size, vocabulary_size)

    #Initialize the hidden layer of the RNN model   
    def initialize_hidden(self,batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    #forward pass while training
    def forward(self, image_features, image_captions):        
        image_captions = image_captions[:, :-1]     
        self.hidden = self.initialize_hidden(self.batch_size) 
                
        embeddings = self.word_embeddings(image_captions)
        embeddings = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        
        lstm_output, self.hidden = self.lstm(embeddings, self.hidden)
        outputs = self.linear(lstm_output)

        return outputs

    #sample the batch
    def sample(self, inputs):        
        data_sample = []
        batch_size = inputs.shape[0]
        hidden_units = self.initialize_hidden(batch_size)
        while True:
            lstm_output, hidden_units = self.lstm(inputs, hidden_units)
            linear_lstm_output = self.linear(lstm_output)
            lstm_outputs = linear_lstm_output.squeeze(1)
            _, max_index = torch.max(lstm_outputs, dim=1)
            data_sample.append(max_index.cpu().numpy()[0].item())
            if (max_index == 1):
                break
            inputs = self.word_embeddings(max_index)
            inputs = inputs.unsqueeze(1)
        return data_sample

    #Getting decoder outputs
    def get_outputs(self, inputs, hidden):
        lstm_output, hidden = self.lstm(inputs, hidden)
        outputs = self.linear(lstm_output)
        outputs = outputs.squeeze(1)
        return outputs, hidden