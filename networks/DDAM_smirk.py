from torch import nn
import torch
from networks import MixedFeatureNet
from torch.nn import Module
import os
import pdb
class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
# class DDAMNet_Smirk(nn.Module):
#     def __init__(self, num_class=8,num_head=2, pretrained=True):
#         super(DDAMNet_Smirk, self).__init__()

#         net = MixedFeatureNet.MixedFeatureNet() # 输入是112*112*3 输出是256 的embediing
                
#         if pretrained:
#             net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))       
#         '''
#         net 是一个 MixedFeatureNet 模型实例。
#         net.children() 返回一个生成器，生成 MixedFeatureNet 模型中的所有子模块。
#         list(net.children()) 将这个生成器转换为一个列表，列表中包含了 MixedFeatureNet 的所有子模块。
#         [:-4] 表示切片操作，去掉列表中的最后四个子模块。但我们主要通过 __init__ 方法中的定义来确定层的顺序和去掉的具体层。
#         * 是解包操作符，将列表中的子模块解包为单独的参数传递给 nn.Sequential。
#         nn.Sequential 将这些子模块按照顺序组合成一个新的模块 self.features。
#         '''
#         self.features = nn.Sequential(*list(net.children())[:-4]) 
#         self.num_head = num_head
#         #CoordAttHead 实例作为属性添加到 DDAMNet 实例中，属性名称使用字符串格式化生成，例如 "cat_head0"、"cat_head1" 等等。
#         for i in range(int(num_head)): #for i in range(int(num_head))：循环遍历从 0 到 num_head-1 的所有整数值，num_head 表示需要创建的 CoordAttHead 实例的数量。
#             setattr(self,"cat_head%d" %(i), CoordAttHead()) #"cat_head%d" % i：属性名称，使用字符串格式化生成，例如 "cat_head0"、"cat_head1" 等等。                 
      
#         self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
#         self.Smirk_Linear = nn.Linear(358, 512, bias=False)
#         self.flatten = Flatten()      
#         self.fusion_layers = nn.Sequential(
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512)
#         )
#         self.fc = nn.Linear(512, num_class)
       
        
#     def forward(self, x,smirk_feature): # linear fusion of the features
#         epsilon = 1e-10
#         #pdb.set_trace()
#         # feature fusion
#         device = x.device
#         x = self.features(x) #特征提取
#         heads = []
       
#         for i in range(self.num_head):
#             heads.append(getattr(self,"cat_head%d" %i)(x)) #，调用每个动态添加的 CoordAttHead 实例，并将其输出添加到 heads 列表中。
#         head_out =heads
        
#         y = heads[0] #将 heads 列表中的第一个元素赋值给 y。

        
#         for i in range(1,self.num_head):
#             y = torch.max(y,heads[i])#通过循环，从第一个元素开始，将 heads 列表中的每个注意力头的输出与 y 进行逐元素比较，取最大值（torch.max），并更新 y。这个过程是为了融合所有注意力头的输出。                     
        
#         y = x*y #将输入特征 x 与融合后的注意力特征 y 逐元素相乘，结合原始特征和注意力特征。#torch.Size([256, 512, 7, 7])
#         #pdb.set_trace()
#         y = self.Linear(y) #torch.Size([256, 512, 1, 1])
#         y = self.flatten(y)#[256,512]  # 在这我们加入新的维度
#        #now  y.shape torch.Size([256, 512])
#         ########### smirk_feature ############
#           ########### 处理 smirk_feature ###########
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]
#         smirk_features = torch.cat(smirk_features_list, dim=1) #torch.Size([256, 358])
#         smirk_features_out = self.Smirk_Linear(smirk_features) #torch.Size([256, 512])
        
#         # 对 y 和 smirk_features_out 进行归一化处理
#         y = (y - y.min()) / (y.max() - y.min()+ epsilon)
#         smirk_features_out = (smirk_features_out - smirk_features_out.min()) / (smirk_features_out.max() - smirk_features_out.min()+ epsilon)

#          # 拼接 y 和 smirk_features_out，形状为 [256, 1024]
#         concatenated_features_y= torch.cat((y, smirk_features_out), dim=1)#
#         # linear feature fusion:
#         fusion_features_y = self.fusion_layers(concatenated_features_y)
#         y = y + fusion_features_y
#         out = self.fc(y) 
        
        
#         return out, x, head_out
    # def forward(self, x, smirk_feature):
    #     epsilon = 1e-8  # 设定一个特别小的数

    #     device = x.device
    #     x = self.features(x)  # 特征提取
    #     heads = []
       
    #     for i in range(self.num_head):
    #         heads.append(getattr(self, "cat_head%d" % i)(x))
    #     head_out = heads
        
    #     y = heads[0]
    #     for i in range(1, self.num_head):
    #         y = torch.max(y, heads[i])
        
    #     y = x * y  # 融合注意力特征和原始特征
    #     y = self.Linear(y)
    #     y = self.flatten(y)
       
    #     smirk_features_list = [
    #         torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
    #     ]
    #     smirk_features = torch.cat(smirk_features_list, dim=1)
    #     smirk_features_out = self.Smirk_Linear(smirk_features)
        
    #     # 对 y 和 smirk_features_out 进行归一化处理
    #     y = (y - y.min()) / (y.max() - y.min()+ epsilon)
    #     smirk_features_out = (smirk_features_out - smirk_features_out.min()) / (smirk_features_out.max() - smirk_features_out.min()+ epsilon)
    #     y_res = y + smirk_features_out
     

    #     out = self.fc_3(y_res) 
        
    #     return out, x, head_out
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
                      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordAtt(512,512)
    def forward(self, x):
        ca = self.CoordAtt(x)
        return ca  
        
class CoordAtt(nn.Module):# 类实现了一种坐标注意力机制，通过同时考虑水平方向和垂直方向的信息来增强输入特征。让我们逐行解析这个类的定义和其前向传播方法：
    def __init__(self, inp, oup, groups=32): #groups：用于分组卷积的组数，默认为 32
        super(CoordAtt, self).__init__()
      
        self.Linear_h = Linear_block(inp, inp, groups=inp, kernel=(1, 7), stride=(1, 1), padding=(0, 0))#self.Linear_h：用于水平方向的卷积，卷积核大小为 (1, 7)。        
        self.Linear_w = Linear_block(inp, inp, groups=inp, kernel=(7, 1), stride=(1, 1), padding=(0, 0))#self.Linear_w：用于垂直方向的卷积，卷积核大小为 (7, 1)。

        
        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()
        self.Linear = Linear_block(oup, oup, groups=oup, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.flatten = Flatten() 

    def forward(self, x):
        # x.shape torch.Size([256, 512, 7, 7])
        identity = x 
        n,c,h,w = x.size()#n, c, h, w = x.size()：获取输入张量的形状，其中 n 是批次大小，c 是通道数，h 是高度，w 是宽度。
        x_h = self.Linear_h(x) # 垂直 #torch.Size([256, 512, 7, 1])
        x_w = self.Linear_w(x) # 水平 #torch.Size([256, 512, 1, 7])
        x_w = x_w.permute(0, 1, 3, 2) #torch.Size([256, 512, 7, 1])
      
        y = torch.cat([x_h, x_w], dim=2)# torch.Size([256, 512, 14, 1])
        y = self.conv1(y)#torch.Size([256, 512, 14, 1])
        y = self.bn1(y)#
        y = self.relu(y) #torch.Size([256, 16, 14, 1])
        x_h, x_w = torch.split(y, [h, w], dim=2) #沿着第二个响亮拆分成h,w -> x_h.shape:torch.Size([256, 512, 7, 1])  x_w.shape:torch.Size([256, 16, 7, 1])
        x_w = x_w.permute(0, 1, 3, 2)#torch.Size([256, 16, 1, 7])

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)#torch.Size([256, 512, 7, 7])
        x_w = x_w.expand(-1, -1, h, w)#torch.Size([256, 512, 7, 7])
     
        y = x_w * x_h #torch.Size([256, 512, 7, 7])
 
        return y
