import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from model.CTA import TA
from thop import profile
from thop import clever_format
import numpy as np


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )



    def forward(self, x,H,W):
        x = x + self.attn(self.norm(x))

        # B, _, C = x.shape
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # y = self.DropBlock(cnn_feat)
        # y = y.flatten(2).transpose(1, 2)
        # y = torch.cat((cls_token.unsqueeze(1), y), dim=1)
        # x = y + self.attn(self.norm(x))
        return x



class DCSPE(nn.Module):
    def __init__(self, dim=512):
        super(DCSPE, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        self.TA=TA(dim)


    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)

        x = self.proj2(self.proj1(self.proj(cnn_feat)))+self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)

        x = self.TA(x)

        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


def standardize(X):
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    return (X - mean) / std



class TDTMIL(nn.Module):
    def __init__(self, n_classes):
        super(TDTMIL, self).__init__()
        self.pos_layer = DCSPE(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes

        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self,x=None, **kwargs):
        if x is not None:
            h = x.float()
        else:
            h = kwargs['data'].float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]


        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x
        h = self.layer1(h,_H, _W) #[B, N, 512]

        #---->DCSPE
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h,_H, _W) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        # print(h.shape)
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict



if __name__ == "__main__":
    data = torch.randn((1, 120, 1024)).cuda()
    model = TDTMIL(n_classes=2).cuda()

    input_data = data

    macs, params = profile(model, inputs=(input_data,))

    macs, params = clever_format([macs, params], "%.3f")

    print(f"Total MACs (FLOPs): {macs}")
    print(f"Total Parameters: {params}")
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)





