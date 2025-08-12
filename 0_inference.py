

import torch 
from model_segmamba.segmamba import SegMamba

t1 = torch.rand(1, 1, 160, 160, 32).cuda()


model = SegMamba(in_chans=1,
                 out_chans=2,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).cuda()

out = model(t1)

print(out.shape)




