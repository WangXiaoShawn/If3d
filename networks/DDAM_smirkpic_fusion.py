from torch import nn
import torch
from networks import MixedFeatureNet
from torch.nn import Module
import os
import pdb
import torch.nn.functional as F

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


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        x1:  (orig_y)
        x2:  (smirk_y)
        """
        batch_size, C, H, W = x1.size()
        proj_query = self.query_conv(x1).view(batch_size, -1, H * W).permute(0, 2, 1)  # [B, N, C']
        proj_key = self.key_conv(x2).view(batch_size, -1, H * W)  # [B, C', N]
        energy = torch.bmm(proj_query, proj_key)  # [B, N, N]
        attention = self.softmax(energy)  # [B, N, N]
        proj_value = self.value_conv(x2).view(batch_size, -1, H * W)  # [B, C, N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(batch_size, C, H, W)
        out = out + x1
        return out
    
class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        batch_size, c, h, w = x.size()
        output = []

        for level in self.levels:
            pooling = F.adaptive_max_pool2d(x, output_size=(level, level))
            output.append(pooling.view(batch_size, -1))

        return torch.cat(output, dim=1)
    
class GateFusion(nn.Module):
    def __init__(self, in_channels):
        super(GateFusion, self).__init__()
        self.gate_fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Sigmoid()  # 激活函数将权重限制在 [0, 1]
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)  # [B, C*2, H, W]
        gate = self.gate_fc(combined)  # [B, C, H, W]
        out = gate * x1 + (1 - gate) * x2  # 门控加权融合
        return out
class DDAMNet_Smirk_Spatial_Alignment_GateFusion(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(DDAMNet_Smirk_Spatial_Alignment_GateFusion, self).__init__()
        orig_feature_net = MixedFeatureNet.MixedFeatureNet() 
        smirk_feature_net = MixedFeatureNet.MixedFeatureNet() 
        if pretrained:
            orig_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            smirk_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))

        self.orig_features = nn.Sequential(*list(orig_feature_net.children())[:-4])
        self.smirk_features = nn.Sequential(*list(smirk_feature_net.children())[:-4])
        self.num_head = num_head

        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())

        self.Linear = Linear_block(512, 512, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.gate_fusion = GateFusion(in_channels=512)

        self.fc1 = nn.Linear(512 * 7 * 7, 2048)  # 注意，这里的尺寸根据卷积后维度调整
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_class + 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, orig_pic, smirk_pic):
        orig_pic = self.orig_features(orig_pic) 
        smirk_pic = self.smirk_features(smirk_pic)
        l3_loss = F.mse_loss(orig_pic, smirk_pic)
        orig_heads = []
        smirk_heads = []

        for i in range(self.num_head):
            orig_heads.append(getattr(self, "cat_head%d" % i)(orig_pic))

        for i in range(self.num_head):
            smirk_heads.append(getattr(self, "cat_head%d" % i)(smirk_pic))

        orig_y = orig_heads[0]
        smirk_y = smirk_heads[0]
        for i in range(1, self.num_head):
            orig_y = torch.max(orig_y, orig_heads[i])
            smirk_y = torch.max(smirk_y, smirk_heads[i])

        orig_y = orig_pic * orig_y
        smirk_y = smirk_pic * smirk_y
        orig_y = self.Linear(orig_y)
        smirk_y = self.Linear(smirk_y)
        fused_y = self.gate_fusion(orig_y, smirk_y)
        y = fused_y.view(fused_y.size(0), -1)  # Flatten
        y = self.leaky_relu(self.bn1(self.fc1(y)))
        y = self.dropout1(y)
        y = self.leaky_relu(self.bn2(self.fc2(y)))
        y = self.dropout2(y)
        y = self.leaky_relu(self.bn3(self.fc3(y)))
        y = self.dropout3(y)
        y = self.leaky_relu(self.bn4(self.fc4(y)))
        output = self.output(y)
        expression = output[:, :8]  
        arousal = output[:, 8]     
        valence = output[:, 9]    

        return expression, orig_heads, smirk_heads, arousal, valence, l3_loss


class DDAMNet_Smirk_Spatial_Alignment_SPPFusion(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(DDAMNet_Smirk_Spatial_Alignment_SPPFusion, self).__init__()
        orig_feature_net = MixedFeatureNet.MixedFeatureNet() 
        smirk_feature_net = MixedFeatureNet.MixedFeatureNet() 
        if pretrained:
            orig_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            smirk_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))

        self.orig_features = nn.Sequential(*list(orig_feature_net.children())[:-4])
        self.smirk_features = nn.Sequential(*list(smirk_feature_net.children())[:-4])
        self.num_head = num_head

        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())

        self.Linear = Linear_block(512, 512, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.spp = SpatialPyramidPooling(levels=[1, 2, 4])

        spp_output_dim = sum([512 * (level * level) for level in self.spp.levels]) * 2

        self.fc1 = nn.Linear(spp_output_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_class + 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, orig_pic, smirk_pic):
        orig_pic = self.orig_features(orig_pic) 
        smirk_pic = self.smirk_features(smirk_pic)
        l3_loss = F.mse_loss(orig_pic, smirk_pic)
        orig_heads = []
        smirk_heads = []

        for i in range(self.num_head):
            orig_heads.append(getattr(self, "cat_head%d" % i)(orig_pic))

        for i in range(self.num_head):
            smirk_heads.append(getattr(self, "cat_head%d" % i)(smirk_pic))

        orig_y = orig_heads[0]
        smirk_y = smirk_heads[0]
        for i in range(1, self.num_head):
            orig_y = torch.max(orig_y, orig_heads[i])
            smirk_y = torch.max(smirk_y, smirk_heads[i])

        orig_y = orig_pic * orig_y
        smirk_y = smirk_pic * smirk_y

        orig_y = self.Linear(orig_y)
        smirk_y = self.Linear(smirk_y)

        orig_spp = self.spp(orig_y)
        smirk_spp = self.spp(smirk_y)

        y = torch.cat((orig_spp, smirk_spp), dim=1)

        y = self.leaky_relu(self.bn1(self.fc1(y)))
        y = self.dropout1(y)
        y = self.leaky_relu(self.bn2(self.fc2(y)))
        y = self.dropout2(y)
        y = self.leaky_relu(self.bn3(self.fc3(y)))
        y = self.dropout3(y)
        y = self.leaky_relu(self.bn4(self.fc4(y)))
        output = self.output(y)
        expression = output[:, :8]  
        arousal = output[:, 8]     
        valence = output[:, 9]    

        return expression, orig_heads, smirk_heads, arousal, valence, l3_loss

class DDAMNet_Smirk_Spatial_Alignment_AttentionFusion(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(DDAMNet_Smirk_Spatial_Alignment_AttentionFusion, self).__init__()
        orig_feature_net = MixedFeatureNet.MixedFeatureNet() 
        smirk_feature_net = MixedFeatureNet.MixedFeatureNet() 
        if pretrained:
            orig_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            smirk_feature_net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))

        self.orig_features = nn.Sequential(*list(orig_feature_net.children())[:-4])
        self.smirk_features = nn.Sequential(*list(smirk_feature_net.children())[:-4])
        self.num_head = num_head

        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.attention_fusion = AttentionFusion(in_channels=512)  
        self.flatten = Flatten()
        self.fc1 = nn.Linear(512 * 1 * 1, 2048) 
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_class + 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, orig_pic, smirk_pic):
        orig_pic = self.orig_features(orig_pic) 
        smirk_pic = self.smirk_features(smirk_pic)
        l3_loss = F.mse_loss(orig_pic, smirk_pic)
        orig_heads = []
        smirk_heads = []

        for i in range(self.num_head):
            orig_heads.append(getattr(self, "cat_head%d" % i)(orig_pic))

        for i in range(self.num_head):
            smirk_heads.append(getattr(self, "cat_head%d" % i)(smirk_pic))

        orig_y = orig_heads[0]
        smirk_y = smirk_heads[0]
        for i in range(1, self.num_head):
            orig_y = torch.max(orig_y, orig_heads[i])
            smirk_y = torch.max(smirk_y, smirk_heads[i])
        orig_y = orig_pic * orig_y
        smirk_y = smirk_pic * smirk_y
        orig_y = self.Linear(orig_y)
        smirk_y = self.Linear(smirk_y)
        y = self.attention_fusion(orig_y, smirk_y)
        y = self.flatten(y)
        y = self.leaky_relu(self.bn1(self.fc1(y)))
        y = self.dropout1(y)
        y = self.leaky_relu(self.bn2(self.fc2(y)))
        y = self.leaky_relu(self.bn3(self.fc3(y)))
        y = self.leaky_relu(self.bn4(self.fc4(y)))
        output = self.output(y)
        expression = output[:, :8]  
        arousal = output[:, 8]     
        valence = output[:, 9]    

        return expression, orig_heads, smirk_heads, arousal, valence, l3_loss
    
    
class DDAMNet_Smirk_Spatial_Alignment(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(DDAMNet_Smirk_Spatial_Alignment, self).__init__()

        net = MixedFeatureNet.MixedFeatureNet() 
                
        if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            
        self.features = nn.Sequential(*list(net.children())[:-4]) 
        self.num_head = num_head
        
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())
        
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.Smirk_Linear = nn.Linear(358, 512, bias=False)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1024, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_class + 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, orig_pic, smirk_pic):

        orig_pic = self.features(orig_pic)  # 特征提取
        smirk_pic = self.features(smirk_pic)
        l3_loss = F.mse_loss(orig_pic, smirk_pic)
        orig_heads = []
        smirk_heads = []
       
        for i in range(self.num_head):
            orig_heads.append(getattr(self, "cat_head%d" % i)(orig_pic))

        for i in range(self.num_head):
            smirk_heads.append(getattr(self, "cat_head%d" % i)(smirk_pic))
        
        
        orig_y = orig_heads[0]
        smirk_y = smirk_heads[0]
        
        for i in range(1, self.num_head):
            orig_y = torch.max(orig_y, orig_heads[i])
            smirk_y = torch.max(smirk_y, smirk_heads[i])
        
        orig_y = orig_pic * orig_y
        smirk_y = smirk_pic * smirk_y
        
        orig_y = self.Linear(orig_y)
        smirk_y = self.Linear(smirk_y)
        
        y = torch.cat([orig_y, smirk_y], dim=1)
        y = self.flatten(y)
        
        y = self.leaky_relu(self.bn1(self.fc1(y)))
        y = self.dropout1(y)
        y = self.leaky_relu(self.bn2(self.fc2(y)))
        # y = self.dropout2(y)
        y = self.leaky_relu(self.bn3(self.fc3(y)))
        y = self.leaky_relu(self.bn4(self.fc4(y)))
        
        output = self.output(y)
        expression = output[:, :8]  # [64,8]
        arousal = output[:, 8]  # [64]
        valence = output[:, 9]  # [64]
        
        return expression, orig_heads, smirk_heads, arousal, valence,l3_loss
    
class DDAMNet_Smirk(nn.Module): 
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(DDAMNet_Smirk, self).__init__()
        net = MixedFeatureNet.MixedFeatureNet()  # 输入是112*112*3 输出是256 的embedding
                
        if pretrained:
            net = torch.load(os.path.join('./pretrained/', "MFN_msceleb.pth"))
            
        self.features = nn.Sequential(*list(net.children())[:-4]) 
        self.num_head = num_head
        
        for i in range(int(num_head)):
            setattr(self, "cat_head%d" % i, CoordAttHead())
        self.Linear = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.Smirk_Linear = nn.Linear(358, 512, bias=False)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1024, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(2048, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_class + 2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, orig_pic, smirk_pic):

        orig_pic = self.features(orig_pic)  # 特征提取
        smirk_pic = self.features(smirk_pic)
        
        orig_heads = []
        smirk_heads = []
       
        for i in range(self.num_head):
            orig_heads.append(getattr(self, "cat_head%d" % i)(orig_pic))

        for i in range(self.num_head):
            smirk_heads.append(getattr(self, "cat_head%d" % i)(smirk_pic))
        
        orig_heads_out = orig_heads
        smirk_heads_out = smirk_heads
        orig_y = orig_heads[0]
        smirk_y = smirk_heads[0]
        
        for i in range(1, self.num_head):
            orig_y = torch.max(orig_y, orig_heads[i])
            smirk_y = torch.max(smirk_y, smirk_heads[i])
        orig_y = orig_pic * orig_y
        smirk_y = smirk_pic * smirk_y
        orig_y = self.Linear(orig_y)
        smirk_y = self.Linear(smirk_y)
        y = torch.cat([orig_y, smirk_y], dim=1)
        y = self.flatten(y)
        y = self.leaky_relu(self.bn1(self.fc1(y)))
        y = self.dropout1(y)
        y = self.leaky_relu(self.bn2(self.fc2(y)))
        y = self.leaky_relu(self.bn3(self.fc3(y)))
        y = self.leaky_relu(self.bn4(self.fc4(y)))
        output = self.output(y)
        expression = output[:, :8] 
        arousal = output[:, 8]
        valence = output[:, 9]
        return expression, orig_heads, smirk_heads, arousal, valence

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
        identity = x 
        n,c,h,w = x.size()#n, c, h, w = x.size()：获取输入张量的形状，其中 n 是批次大小，c 是通道数，h 是高度，w 是宽度。
        x_h = self.Linear_h(x) # 垂直 #torch.Size([256, 512, 7, 1])
        x_w = self.Linear_w(x) # 水平 #torch.Size([256, 512, 1, 7])
        x_w = x_w.permute(0, 1, 3, 2) #torch.Size([256, 512, 7, 1])
      
        y = torch.cat([x_h, x_w], dim=2)# torch.Size([256, 512, 14, 1])
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2) 
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = x_w * x_h 
 
        return y
