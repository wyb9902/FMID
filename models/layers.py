import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# # AttBlock
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        # #
        self.my1 = myconv1(out_channel) if filter else nn.Identity()
        self.my2 = myconv2(out_channel) if filter else nn.Identity()


    
        self.proj = nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel)
        self.proj_act = nn.GELU()


    def forward(self, x):

        out = self.conv1(x)

        out = self.proj(out)
        out = self.proj_act(out)
        out = self.my1(out)
        out = self.my2(out)
        out = self.conv2(out)
        
        return out + x

##FECA
class myconv1(nn.Module):
    def __init__(self, planes, pooling_r=4, kernel=3) -> None:
        super(myconv1, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes)
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1)
        )
        self.k4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, kernel * planes, kernel_size=1, stride=1, bias=False, groups=1),
            nn.BatchNorm2d(kernel * planes),
            nn.Tanh()
        )
        self.kernel = (1, kernel)
        pad_r = pad_l = kernel // 2
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        self.gamma = nn.Parameter(torch.zeros(planes, 1, 1))
        self.beta = nn.Parameter(torch.ones(planes, 1, 1))

    def forward(self, x):
        identity = x

        filter = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        filter = torch.mul(self.k3(x), filter)  # k3 * sigmoid(identity + k2)
        filter = self.k4(filter)  # k4
        b, c, h, w = filter.shape

        filter = filter.view(b, self.kernel[1], c // self.kernel[1], h * w).permute(0, 1, 3, 2).contiguous()
        # print(filter.size())
        B, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).view(B, H * W, C).unsqueeze(1)
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1)
        out = out.view(B, self.kernel[1], H * W, -1)
        out = torch.sum(out * filter, dim=1, keepdim=True).permute(0, 3, 1, 2).reshape(B, C, H, W)

        return out * self.gamma + x * self.beta


######
##FCSA
class myconv2(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features/r), L)
        self.features = features

        self.conv1 = nn.Sequential(
            nn.Conv2d(features*2, features, 1),
            nn.GELU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(features, features, 1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, features // 2, 1),
            nn.GELU()
        )


        self.convh = nn.Sequential(
            nn.Conv2d(features, features,  kernel_size=5, stride=1, padding=4, groups=features, dilation=2),
            nn.GELU()
        )

        self.convl = nn.Sequential(
            # nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1,groups=features),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.fc = nn.Sequential(
            nn.Conv2d(features, d, 1, 1, 0),
            nn.BatchNorm2d(d),
            nn.GELU()
        )

        # self.fc2 = nn.Conv2d(d, features, 1, 1, 0)
        self.fc2 = nn.ModuleList([])
        for i in range(M):
            self.fc2.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros((1,features,1,1)), requires_grad=True)
        self.c221 = nn.Conv2d(2, 1, 1, 1, 0)

    def forward(self, x):
        ##
        h = x.size(2)
        w = x.size(3)
        low = self.convl(x)
        high = self.convh(low)
        emerge = torch.cat([high, low], dim=1)
        avg_attn = torch.mean(emerge, dim=1, keepdim=True)
        max_attn, _ = torch.max(emerge, dim=1, keepdim=True)
        agg = torch.cat([max_attn, avg_attn], dim=1)
        agg = self.c221(agg)
        emerge = self.conv1(emerge)

        gw = nn.AvgPool2d(kernel_size=(1, w), stride=(1, w))
        gh = nn.AvgPool2d(kernel_size=(h, 1), stride=(h, 1))
        emergeh = gh(emerge)
        emergew = gw(emerge)


        emergeh = self.conv11(emergeh)
        emergew = self.conv11(emergew)

        high_att = self.fc(emergeh)
        high_att = self.fc2[0](high_att)

        low_att = self.fc(emergew)
        low_att = self.fc2[1](low_att)

        high_att = self.softmax(high_att)
        low_att = self.softmax(low_att)

        w1 = high_att * agg
        w2 = low_att * agg

        fea_high = high * w1

        fea_low = low * w2

        out = self.out(fea_high + fea_low)
        return out * self.gamma + x
