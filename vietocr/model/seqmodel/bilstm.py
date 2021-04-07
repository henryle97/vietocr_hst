import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_out, n_hidden):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.fc_out = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, B, H = recurrent.size()
        t_rec = recurrent.view(T*B, H)

        output = self.fc_out(t_rec) # [T*b, n_out]
        output = output.view(T, B, -1)
        return output

class RNN_layer(nn.Module):
    def __init__(self,  n_in, n_class, n_hidden):
        super(RNN_layer, self).__init__()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(n_in, n_hidden, n_hidden),
            BidirectionalLSTM(n_hidden, n_hidden, n_class)
        )

    def forward(self, x):
        out = self.rnn(x)
        out = F.log_softmax(out, dim=2)
        return out

