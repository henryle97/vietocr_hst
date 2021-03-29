import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    '''
    Cross Entropy + Smoothing Loss
    CrossEntropy Loss: L = 1/N * sum(-label * log(softmax(output))
    Smoothing: label: 1/0 -->
    '''


    def __init__(self, classes, padding_idx, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.padding_idx = padding_idx

    def forward(self, pred, target):
        '''

        Args:
            pred: B*tgt_len x n_class
            target: B*tgt_len

        Returns:

        '''
        pred = pred.log_softmax(dim=self.dim)     # log + softmax
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 2))      # exclude: <SOS> and <EOS>
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  #scatter_(dim, indexs, value)
            true_dist[:, self.padding_idx] = 0                                     # remove padding in outputs
            mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)  # remove padding in target
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))   #



