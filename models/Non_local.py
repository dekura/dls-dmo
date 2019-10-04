import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)

class Mean(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(x*y, dim=1, keepdim=True)


class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))
        self.bmm = BMM()

        self.feat = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, _BatchNorm):
        #         m.weight.data.fill_(1)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):

        x = self.feat(x)
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = self.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = self.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=nn.BatchNorm2d):
        super(EncModule, self).__init__()
        # norm_layer = nn.BatchNorm1d if isinstance(norm_layer, nn.BatchNorm2d) else \
        #     encoding.nn.BatchNorm1d
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            # encoding.nn.Encoding(D=in_channels, K=ncodes),
            # norm_layer(ncodes),
            # nn.ReLU(inplace=True),
            # encoding.nn.Mean(dim=1)
            )

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en.view(b,-1))
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)

class RecoHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,  dim=512,  se_loss=False):
        super(RecoHead, self).__init__()
        inter_channels = in_channels // 4
        h = dim // 8
        # self.decomp = ContextDecomp(in_channels, h, norm_layer, up_kwargs)

        self.decomp =ContextDecomp(h, norm_layer=nn.BatchNorm2d)
        # self.conv5 = nn.Sequential(
        #                            nn.Conv2d(512 + 512, inter_channels, 3, padding=1, bias=False),
        #                            norm_layer(inter_channels),
        #                            nn.ReLU(True),
        #                            )
        self.encmodule = EncModule(512, out_channels, ncodes=32,
                                   se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(
                                   nn.Conv2d(512*2, inter_channels, 1, padding=0, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1),
                                   )

        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, dilation=1, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True))

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.bmm = BMM()
    def forward(self, x):
        # att = x
        # b, c, h, w = att.size()
        # Q = self.query_conv(att).view(b, -1, h * w).permute(0, 2, 1)
        # K = self.key_conv(att).view(b, -1, h * w)
        # energy = self.bmm(Q, K)
        # attention = self.softmax(energy)
        # V = self.value_conv(att).view(b, -1, h * w)
        # Vout = self.bmm(V, attention.permute(0, 2, 1))
        # Vout = Vout.view(b, c, h, w)
        # att = (att + Vout)

        feat = self.feat(x)
        feat_outs = list(self.encmodule(feat))
        outs = self.decomp(feat)
        feat_outs[0] = self.conv6(torch.cat((outs, feat_outs[0]), 1))
        return feat_outs

class ContextDecomp(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, h, norm_layer=nn.BatchNorm2d):
        super(ContextDecomp, self).__init__()
        self.rank = 128
        self.ps = [1, 1, 1, 1]
        self.h = h
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, h)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])

        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=0, bias=False),
            norm_layer(512),
            # nn.Sigmoid(),
            nn.ReLU(True),
        )

        # self.pre_fusion = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, dilation=1, padding=1, bias=False),
        #     norm_layer(512),
        #     # # nn.Sigmoid(),
        #     nn.ReLU(True),
        # )
        self.BMM = BMM()



    def forward(self, x):
        # x = self.pre_fusion(x)
        b, c, height, width = x.size()
        C = self.pool(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())
        list = []
        for i in range(0, self.rank):
            list.append(self.TukerReconstruction(b, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](H), self.conv1_3[i](W)))
        tensor1 = sum(list)*0.015625
        tensor1 = x + x * tensor1
        tensor1 = self.fusion(tensor1)
        # y = torch.cat((x , x * tensor1), 1)
        return tensor1

    def ConvGeneration(self, rank, h):
        conv1 = []
        n = 1
        for _ in range(0, rank):
                conv1.append(nn.Sequential(
                nn.Conv2d(512, 512 // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
                conv2.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
                conv3.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TukerReconstruction(self, batch_size, h, ps, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, ps)
        H = feat2.view(b, ps, -1)
        W = feat3.view(b, ps * ps, -1)
        CHW = self.BMM(self.BMM(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)
        return CHW

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

class SeparableATT(nn.Module):
    def __init__(self, spatial=16, out_channel=256):
        super(SeparableATT, self).__init__()
        self.out_channel = out_channel
        self.spatial = spatial

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.Mean = Mean()

        # w = torch.Tensor(64*64, 64*64 // 64, 1, 1)
        # nn.init.kaiming_normal_(w, mode='fan_out')
        # self.pw_weight = nn.Parameter(w, requires_grad=True)
        self.se_conv = nn.Sequential(
            nn.Conv2d(in_channels=64*64, out_channels=64*64 // 64, kernel_size=1),
            nn.BatchNorm2d(64*64 // 64),
            nn.ReLU(True),
            nn.Softmax(dim=1),
            nn.Conv2d(in_channels=64 * 64 // 64, out_channels=64 * 64, kernel_size=1),
        )

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        #####Depthiwse-Attention#############
        Q = self.query_conv(x)
        K = self.key_conv(x)
        attention_dw = self.Mean(Q,K)
        attention_dw = F.sigmoid(attention_dw)
        V = self.value_conv(x)
        V_dw = V * attention_dw
        ######Pointwise-Attention#############
        V_pw = V_dw.view(b, -1, h * w).permute(0, 2, 1)
        V_pw = V_pw.view(b, h * w, self.spatial, c // self.spatial)
        V_pw = self.se_conv(V_pw)

        # pw_weight = F.softmax(self.pw_weight, dim=1)
        # Vout = F.conv2d(V_pw, pw_weight)
        Vout = V_pw

        Vout = Vout.view(b, h * w, c).permute(0, 2, 1)
        Vout = Vout.view(b, c, h, w)

        att = (x + Vout)
        return att

class SeparableATT2(nn.Module):
    def __init__(self, spatial=16, out_channel=256, norm_layer=None):
        super(SeparableATT2, self).__init__()
        self.out_channel = out_channel
        self.spatial = spatial

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.Mean = Mean()
        self.se_conv = nn.Sequential(
            nn.Conv2d(in_channels=64*64, out_channels=64*64 // 16, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64 * 64 // 16, out_channels=64 * 64, kernel_size=1),
            nn.Sigmoid(),
        )

        # self.gamma = nn.Parameter(torch.ones(out_channels))
        # print(self.pw_weight)

    def forward(self, x):
        b, c, h, w = x.size()

        #####Depthiwse-Attention#############
        Q = self.query_conv(x)
        K = self.key_conv(x)
        attention_dw = self.Mean(Q, K)
        attention_dw = attention_dw.view(b, 1, h * w).permute(0, 2, 1)
        attention_dw = torch.unsqueeze(attention_dw, 2)
        attention_dw = self.se_conv(attention_dw)
        attention_dw = attention_dw.view(b, h * w, 1).permute(0, 2, 1)
        attention_dw = attention_dw.view(b, 1, h, w)
        V = self.value_conv(x)
        Vout = V * attention_dw

        att = (x + Vout)
        return att

class NonLocalATT(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=512, out_channel=256):
        super(NonLocalATT, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.batch_matmul = BMM()

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # proj_key = proj_query.permute(0, 2, 1)

        energy = self.batch_matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = self.batch_matmul(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = out + x
        return out

class SeparableATT4(nn.Module):
    def __init__(self, spatial=16, out_channel=256, dim=64, norm_layer=None):
        super(SeparableATT4, self).__init__()
        self.out_channel = out_channel
        self.spatial = spatial

        # self.project = nn.Sequential(
        #     nn.Conv2d(512, 512, 1, bias=False),
        #     norm_layer(512),
        #     nn.ReLU(inplace=True)
        # )

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=self.out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)

        self.squeeze_conv = nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1)
        self.extraction_conv = nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1)

        self.bmm=BMM()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        # x = self.project(x)
        b, c, h, w = x.size()

        Q = self.query_conv(x)
        K = self.key_conv(x)
        attention_dw = torch.mean(Q * K, dim=1, keepdim=True)
        attention_dw = F.sigmoid(attention_dw)
        V = self.value_conv(x)
        V_dw = V * attention_dw
        V_dw = V_dw.view(b, -1, h * w)

        Q = self.squeeze_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        K = self.extraction_conv(x).view(b, -1, h * w)

        energy = self.bmm(V_dw, Q)
        attention = self.softmax(energy)
        Vout = self.bmm(attention, K)
        Vout = Vout.view(b, c, h, w)
        att = (x + Vout)

        return att


class APNB(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=512, out_channel=256, psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.psp = PSPModule(psp_size)
        self.batch_matmul = BMM()

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.psp(self.key_conv(x))

        energy = self.batch_matmul(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.psp(self.value_conv(x))

        out = self.batch_matmul(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = out + x
        return out


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.conv1 = APNB(in_dim=512, out_channel=256)

    def forward(self, x):
        x = self.conv1(x)
        return x

class NL(nn.Module):
    def __init__(self):
        super(NL, self).__init__()
        # self.conv1 = SeparableATT(spatial=16, out_channel=256)
        self.conv1 = NonLocalATT(in_dim=512, out_channel=256)
        # self.conv1 = APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256, dropout=0.05)

    def forward(self, x):
        x = self.conv1(x)
        return x

class SNL(nn.Module):
    def __init__(self):
        super(SNL, self).__init__()
        self.conv1 = SeparableATT4(spatial=1, out_channel=256)

    def forward(self, x):
        x = self.conv1(x)
        return x

class SNL2(nn.Module):
    def __init__(self):
        super(SNL2, self).__init__()
        self.conv1 = SeparableATT2(spatial=1, out_channel=256)


    def forward(self, x):
        x = self.conv1(x)
        return x

class RecoNet(nn.Module):
    def __init__(self):
        super(RecoNet, self).__init__()
        # self.conv1 = RecoHead( in_channels=512, out_channels=21)
        self.conv1 =  ContextDecomp(h=64, norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        x = self.conv1(x)
        return x

class EMANet(nn.Module):
    def __init__(self):
        super(EMANet, self).__init__()
        self.conv1 = EMAU(512, 64, 3)


    def forward(self, x):
        x = self.conv1(x)
        return x

# if __name__ == "__main__":
#     device = torch.device("cuda:0")
#     x = Variable(torch.randn(1, 512, 64, 64), requires_grad=True).to(device)
#     net1 = ANN().to(device)
#     stat(ANN(), (512, 64, 64))
#     net2 = NL().to(device)
#     stat(NL(), (512, 64, 64))
#     net3 = SNL().to(device)
#     stat(SNL(), (512, 64, 64))
    # net4 = SNL2().to(device)
    # stat(SNL2(), (512, 64, 64))
    # net5 = RecoNet().to(device)
    # stat(RecoNet(), (512, 64, 64))
    # net6 = EMANet().to(device)
    # stat(EMANet(), (512, 64, 64))
#     with torch.autograd.profiler.profile(use_cuda=True) as prof1:
#         for i in range(100):
#             y = net1(x)
# # NOTE: some columns were removed for brevity
# #         print(prof1.key_averages().table(sort_by="cuda_time_total"))
#     print('#############Asymmetric-NL#############')
#     print(prof1.total_average())
#     with torch.autograd.profiler.profile(use_cuda=True) as prof2:
#         for i in range(100):
#             y = net2(x)
#     # NOTE: some columns were removed for brevity
#     # print(prof2.key_averages().table(sort_by="cuda_time_total"))
#     print('#############Non-Local#############')
#     print(prof2.total_average())
#     with torch.autograd.profiler.profile(use_cuda=True) as prof3:
#         for i in range(100):
#             y = net3(x)
#     # NOTE: some columns were removed for brevity
#     # print(prof3.key_averages().table(sort_by="cuda_time_total"))
#     print('#############Separable-NL#############')
#     print(prof3.total_average())
#     with torch.autograd.profiler.profile(use_cuda=True) as prof4:
#         for i in range(100):
#             y = net4(x)
#     # NOTE: some columns were removed for brevity
#     # print(prof4.key_averages().table(sort_by="cuda_time_total"))
#     print('#############Separable-NL2#############')
#     print(prof4.total_average())


# torch.cuda.synchronize()
# start = time.time()
# result = model(input)
# torch.cuda.synchronize()
# end = time.time()

