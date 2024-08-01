import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    reference: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class FocalLossWithLabelSmooth(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, eps=0.1):
        '''
        Formula:
        label_smooth = (1 - α) * y_hot + α / classes
        focal_loss = -alpha*((1-p)^gamma)*log(p)
        '''
        super(FocalLossWithLabelSmooth, self).__init__()
        # for label smooth
        self.eps = eps
        # for focal loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        class_num = pred.size(1)
        label = label.contiguous().view(-1)
        # for label smooth
        one_hot_label = torch.zeros_like(pred)
        one_hot_label = one_hot_label.scatter(1, label.view(-1, 1), 1)
        one_hot_label = one_hot_label * (1 - self.eps) + (1 - one_hot_label) * self.eps / (class_num-1)
        # print(one_hot_label) 
        log_prob = F.log_softmax(pred, dim=1)
        CEloss = (one_hot_label * log_prob).sum(dim=1)
        # for focal loss
        P = F.softmax(pred, 1)
        class_mask = pred.data.new(pred.size(0), pred.size(1)).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        probs = (P * class_mask).sum(1).view(-1, 1)
        # print(probs) 
        # if multi-class you need to modify here 
        alpha = torch.empty(label.size()).fill_(1 - self.alpha)
        # TODO: multi class
        alpha[label == 1] = self.alpha                                                                                                                     
        
        if pred.is_cuda and not alpha.is_cuda:                                                                                                             
            alpha = alpha.cuda()                                                                                                                           
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * CEloss                                                                                
        loss = batch_loss.mean()                                                                                                                           
        return loss   




if __name__ == '__main__':
    torch.manual_seed(1)
    inputs = Variable(torch.randn((10, 2)))
    targets = Variable(torch.LongTensor(10).random_(2))
    loss = FocalLoss()(inputs, targets)
    print(loss)