
# location for defining the Pytorch model architecture that will be read in during model training

import torch
import torch.nn as nn
import torch.nn.functional as F


# a version of the model for hyperparameter optimisation, that has the opportunity to change the model hyperparameters on init
class LSTM_Dropout(nn.Module):

    def __init__(self, num_input_features, hidden_size, input_dropout, linear_dropout, linear2_in, linear3_in,
                 activation_function="leaky_relu"):
        super(LSTM_Dropout, self).__init__()
        self.input_size = num_input_features
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.input_dropout = input_dropout
        self.linear_dropout = linear_dropout
        self.bidirectional = True
        self.linear2_in = linear2_in
        self.linear3_in = linear3_in

        if activation_function == "leaky_relu": self.activation_function = F.leaky_relu
        elif activation_function == "relu": self.activation_function = F.relu
        elif activation_function == "sigmoid": self.activation_function = torch.sigmoid
        elif activation_function == "tanh": self.activation_function = torch.tanh
        else:
            raise ValueError("Activation function not recognised, currently only supports leaky_relu, relu, sigmoid, and tanh")

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)

        # if using a bidirectional LSTM, the input to the first linear layer will be twice the hidden size
        if self.bidirectional: self.linear1_in = self.hidden_size * 2
        else: self.linear1_in = self.hidden_size

        # define linear layers
        self.linear1 = nn.Linear(self.linear1_in, self.linear2_in)
        self.linear2 = nn.Linear(self.linear2_in, self.linear3_in)
        self.linear3 = nn.Linear(self.linear3_in, 1)

        # define dropout layer
        self.input_dropout_layer = nn.Dropout(p=self.input_dropout)  # dropout layer before input is passed to LSTM
        self.linear_dropout_layer = nn.Dropout(p=self.linear_dropout)  # dropout layer after LSTM, and after each linear layer


    def forward(self, x):

        self.sequence_length = x.shape[1]

        batch_size = x.size(0)  # need the batch size for reshaping the output

        x = self.input_dropout_layer(x)  # apply the first dropout layer to inputs - trying to prevent overfitting to certain inputs

        lstm_out, _ = self.lstm(x)  # pass the input through the LSTM layer

        lstm_out = self.linear_dropout_layer(lstm_out) # dropout layer before linear layers

        # Reshape the output tensor to combine the batch and sequence dimensions
        lstm_out = lstm_out.contiguous().view(-1, self.linear1_in)

        # pass the flattened output through the linear layers
        out = self.activation_function(self.linear1(lstm_out))  # rectified linear unit activation function
        out = self.linear_dropout_layer(out) # dropout layer
        out = self.activation_function(self.linear2(out))  # rectified linear unit activation function
        out = self.linear_dropout_layer(out) # dropout layer
        out = self.linear3(out)

        out = out.view(batch_size, self.sequence_length, 1)  # reshape the output to be batch first

        return out
    
    
    
