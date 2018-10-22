import torch.nn as nn
import torch

alpha = 3

def accuracy(preds, labels, top=[1,5]):
    """
    Computes the precision@top for the specified values of top
    Import from  https://github.com/labyrinth7x/JointOptimization/blob/master/utils/criterion.py #L24-L36
    """
    result = []
    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted labels for all samples in the mini-batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    for k in top:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result


def joint_loss(preds_labeled, preds_unlabeled, labels, epoch, gpu):
    train_criterion = LogisticLoss(gpu)
    loss_unlabeled = 0.0
    if preds_unlabeled is not None:
        pseudo_labels = preds_unlabeled.argmax(dim = 1)
        loss_unlabeled = train_criterion(preds_unlabeled, pseudo_labels)
        if epoch >= T2:
            weight = alpha
        else:
            weight = alpha * (epoch - T1) / (T2 - T1)
        loss_unlabeled = weight * loss_unlabeled / preds_unlabeled.size(0)
    
    loss_labeled = train_criterion(preds_labeled, labels) / labels.size(0)
    loss = loss_labeled + loss_unlabeled
    return loss
    

class LogisticLoss(nn.Module):
    def __init__(self, gpu):
        super(LogisticLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction=None)
        self.gpu = gpu

    def forward(self, preds, labels):
        labels_sigmoid = torch.zeros(preds.size())
        for index in range(len(labels)):
            labels_sigmoid[index][int(labels[index])] = 1

        labels_sigmoid = labels_sigmoid.cuda(self.gpu, non_blocking=True)

        loss = self.criterion(preds, labels_sigmoid)
        return loss
