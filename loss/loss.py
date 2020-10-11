import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMCL(nn.Module):

    def __init__(self, in_features, out_features,s=None, m=None):

        super(LMCL, self).__init__()
        self.s = 64.0 if not s else s
        self.m = 1.35 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, labels):

        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        self.fc.weight=torch.nn.Parameter(self.fc.weight / torch.norm(self.fc.weight, dim=1, keepdim=True))


        x = F.normalize(x,p=2,dim=1)

        wf = self.fc(x)

        self.b_y=0
        self.b_i=0
        if self.fc.bias is not None:
            self.b=self.fc.bias
            self.b=torch.nn.Parameter(self.b.expand(len(labels),len(self.b)))
            self.b_i=torch.cat([torch.cat((self.b[i, :y], self.b[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            self.b_y=torch.cat([self.b[i, y].unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)+self.b_y
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * (excl)+self.b_i), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).cuda())


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


