
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import os
import sys


from model.pv_conv_utils import *




class AllGloskip(nn.Module):
    #  num_croblock=[0,0,0,0]
    def __init__(self, in_chan, base_chan, num_classes=9, num_croblock=[0, 0, 2, 0],embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], attn_drop=0., drop_path_rate=0.1,bottleneck=False,
                 maxpool=True,interplot = True, mode="train", depths=[2,2,9,3],deep_supervision=False):
        super().__init__()

        ## 卷积提取特征stage1
        self.inc = [BasicBlock(in_chan, base_chan), BasicBlock(base_chan, base_chan)]
        self.mode = mode
        self.deep_supervision = deep_supervision

        self.inc = nn.Sequential(*self.inc)
        self.depths = depths
        self.embed_dims = embed_dims

        # pyramid pooling ratios for each stage
        pool_ratios = [[12, 16, 20, 24,56], [6, 8, 10, 12,28], [3, 4, 5, 6,14], [1, 2, 3, 4,7]]



        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # 下采样
        # num_block决定卷积block数
        self.down1 = Glo_tkdown_block_pv(base_chan, embed_dims=embed_dims[0], bottleneck=bottleneck, maxpool=maxpool,
                                   heads=num_heads[0], num_croblock=num_croblock[0], depths=depths[0],cur=cur,
                                       attn_drop=attn_drop, dpr=dpr,mlp_ratios=mlp_ratios[0],pool_ratios=pool_ratios[0])
        cur += depths[0]
        self.down2 = Glo_tkdown_block_pv(embed_dims[0], embed_dims=embed_dims[1], bottleneck=bottleneck, maxpool=maxpool,
                                   heads=num_heads[1], num_croblock=num_croblock[1], depths=depths[1],cur=cur,
                                   attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[1],
                                   pool_ratios=pool_ratios[1])
        cur += depths[1]
        self.down3 = Glo_tkdown_block_pv(embed_dims[1], embed_dims=embed_dims[2], bottleneck=bottleneck, maxpool=maxpool,
                                   heads=num_heads[2], num_croblock=num_croblock[2], depths=depths[2],cur=cur,
                                   attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[2],
                                   pool_ratios=pool_ratios[2])
        cur += depths[2]
        self.down4 = Glo_tkdown_block_pv(embed_dims[2], embed_dims=embed_dims[3], bottleneck=bottleneck, maxpool=maxpool,
                                   heads=num_heads[3], num_croblock=num_croblock[3], depths=depths[3],cur=cur,
                                   attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[3],
                                   pool_ratios=pool_ratios[3])
        cur =0

        self.up1 = skip_block(embed_dims[3], embed_dims=embed_dims[2], bottleneck=bottleneck, interplot=interplot,
                                   heads=num_heads[2], num_croblock=num_croblock[2], depths=depths[2], cur=cur,
                                   attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[2],
                                   pool_ratios=pool_ratios[2])
        cur += depths[2]
        self.up2 = skip_block(embed_dims[2], embed_dims=embed_dims[1], bottleneck=bottleneck, interplot=interplot,
                                 heads=num_heads[1], num_croblock=num_croblock[1], depths=depths[1], cur=cur,
                                 attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[1],
                                 pool_ratios=pool_ratios[1])
        cur += depths[1]
        self.up3 = skip_block(embed_dims[1], embed_dims=embed_dims[0], bottleneck=bottleneck, interplot=interplot,
                                 heads=num_heads[0], num_croblock=num_croblock[0], depths=depths[0], cur=cur,
                                 attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[0],
                                 pool_ratios=pool_ratios[0])
        cur += depths[0]
        self.up4 = skip_block(embed_dims[0], embed_dims=base_chan, bottleneck=bottleneck, interplot=interplot,
                                 heads=1, num_croblock=0, depths=depths[1], cur=cur,
                                 attn_drop=attn_drop, dpr=dpr, mlp_ratios=mlp_ratios[0],
                                 pool_ratios=pool_ratios[0])

        if self.deep_supervision:
            self.final4 = Dsv_UP(in_size=embed_dims[2],out_size = num_classes, scale_factor=8)
            self.final3 = Dsv_UP(in_size=embed_dims[1],out_size = num_classes, scale_factor=4)
            self.final2 = Dsv_UP(in_size=embed_dims[0],out_size = num_classes, scale_factor=2)
            self.final = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)
            self.out = nn.Conv2d(num_classes*4, num_classes, 1)
            # self.final4 = Dsv_UP(in_size=embed_dims[2],out_size = num_classes, scale_factor=8)
        else:
            self.out = nn.Conv2d(base_chan, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outlist = []
        outren = []
        x = self.inc(x)

        x1 = x  #torch.Size([3, 32, 224, 224])
        x2 = self.down1(x1)  # torch.Size([3, 64, 112, 112])

        x3 = self.down2(x2)  # torch.Size([3, 128, 56, 56])
        outlist.append(x3)
        x4 = self.down3(x3)  # torch.Size([3, 256, 28, 28])
        outlist.append(x4)
        x5 = self.down4(x4)   # # torch.Size([3, 512, 14, 14])
        outlist.append(x5)

        out4 = self.up1(x5, x4)
        out3 = self.up2(out4, x3)
        out2 = self.up3(out3, x2)  # torch.Size([3, 64, 112, 112])
        out1 = self.up4(out2, x1)  # torch.Size([3, 64, 224, 224])

        if self.deep_supervision:
            dsv4 = self.final4(out4)
            dsv3 = self.final3(out3)
            dsv2 = self.final2(out2)
            dsv1 = self.final(out1)
            dv = self.out(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))
            return [dsv4,dsv3,dsv2,dsv1,dv]
        else:
            # outlist.append(out)
            out = self.out(out1)
            # outlist.append(out)
            return out



        

        
