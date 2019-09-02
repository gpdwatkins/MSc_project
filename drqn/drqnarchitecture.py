import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQNNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_dim, no_layers, seed = None, fc1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DRQNNetwork, self).__init__()

        self.no_layers = no_layers
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers

        if not seed is None:
            self.seed = torch.manual_seed(seed)

        self.lstm = nn.LSTM(state_size, hidden_dim, no_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state, hidden):
        """Build a network that maps state -> action values."""
        # print(hidden[0].shape)
        lstm_out, hidden = self.lstm(state, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        x = F.relu(self.fc1(lstm_out))
        x = F.relu(self.fc2(x))
        return self.fc3(x), hidden

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.no_layers, batch_size, self.hidden_dim).zero_().to(device),
    #                   weight.new(self.no_layers, batch_size, self.hidden_dim).zero_().to(device))
    #     return hidden

#
# class SentimentNet(nn.Module):
#     def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
#         super(SentimentNet, self).__init__()
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
#         self.dropout = nn.Dropout(drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, hidden):
#         batch_size = x.size(0)
#         x = x.long()
#         embeds = self.embedding(x)
#         lstm_out, hidden = self.lstm(embeds, hidden)
#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
#
#         out = self.dropout(lstm_out)
#         out = self.fc(out)
#         out = self.sigmoid(out)
#
#         out = out.view(batch_size, -1)
#         out = out[:,-1]
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                       weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#         return hidden
