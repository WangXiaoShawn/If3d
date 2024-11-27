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
    
# class SmirkNet(nn.Module):  # 50 feature 
#     def __init__(self, num_class=8,num_head=2, pretrained=True):
#         super(SmirkNet, self).__init__()
#         self.Smirk_Linear = nn.Linear(55, 512)

#         self.linear_layers = nn.Sequential(
#             nn.Linear(512,1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 512)
#         )
#         self.fc = nn.Linear(512, num_class)
       
        
#     def forward(self,x,smirk_feature): # linear fusion of the features
#         epsilon = 1e-10
#         device = x.device
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]
#         #pdb.set_trace()
        
#         smirk_features = torch.cat(smirk_features_list, dim=1) #torch.Size([256, 55])
#         smirk_features_out = self.Smirk_Linear(smirk_features) #torch.Size([256, 512])
#         smirk_features_out= self.linear_layers(smirk_features_out)
#         out = self.fc(smirk_features_out) #torch.Size([256, 8])        
        
#         return out
        
# class SmirkNet(nn.Module): # 简单的线性层 输入358 json file
#     def __init__(self, num_class=8,num_head=2, pretrained=True):
#         super(SmirkNet, self).__init__()
#         self.Smirk_Linear = nn.Linear(358, 512)

#         self.linear_layers = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 512)
#         )
#         self.fc = nn.Linear(512, num_class)
       
        
#     def forward(self,x,smirk_feature): # linear fusion of the features
#         epsilon = 1e-10
#         device = x.device
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]

#         smirk_features = torch.cat(smirk_features_list, dim=1) #torch.Size([256, 358])
#         smirk_features_out = self.Smirk_Linear(smirk_features)  #torch.Size([256, 512])
#         smirk_features_out= self.linear_layers(smirk_features_out) 
#         out = self.fc(smirk_features_out) #torch.Size([256, 8])
    
        
#         return out
class SmirkNet(nn.Module):
    def __init__(self, num_class=8, num_head=2, pretrained=True):
        super(SmirkNet, self).__init__()
        self.Smirk_Linear = nn.Linear(358, 512)

        self.residual_block1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )

        self.residual_block2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )

        self.fc = nn.Linear(512, num_class)
        
    def forward(self, x, smirk_feature):  # residual connection #输入358 json file
        epsilon = 1e-10
        device = x.device
        smirk_features_list = [
            torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
        ]

        smirk_features = torch.cat(smirk_features_list, dim=1)  # torch.Size([256, 358])
        smirk_features_out = self.Smirk_Linear(smirk_features)  # torch.Size([256, 512])

        # Residual connection for the first block
        residual = smirk_features_out
        smirk_features_out = self.residual_block1(smirk_features_out)
        smirk_features_out += residual

        # Residual connection for the second block
        residual = smirk_features_out
        smirk_features_out = self.residual_block2(smirk_features_out)
        smirk_features_out += residual

        out = self.fc(smirk_features_out)  # torch.Size([256, 8])
        
        return out
# class SmirkNet(nn.Module):
#     def __init__(self, num_class=8, num_head=1, dropout_rate=0.1):
#         super(SmirkNet, self).__init__()
#         self.bn1 = nn.BatchNorm1d(358)
        
#         self.self_attention = nn.MultiheadAttention(embed_dim=1, num_heads=num_head, dropout=dropout_rate, batch_first=True)
    
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(358, num_class)
       
#     def forward(self, x, smirk_feature): 
#         device = x.device
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]
        
#         smirk_features = torch.cat(smirk_features_list, dim=1)  # torch.Size([batch_size, 358])
        
#         # Apply batch normalization
#         smirk_features_n = self.bn1(smirk_features) # torch.Size([256, 358])
#         smirk_features = smirk_features_n.unsqueeze(2) # torch.Size([256, 358, 1])
#         #pdb.set_trace()
#         # Apply self-attention
#         attn_output, _ = self.self_attention(smirk_features, smirk_features, smirk_features) # torch.Size([256, 358, 1]) #query, key, value
#         attn_output = attn_output.squeeze(2) # torch.Size([256, 358])
#         attn_output = self.dropout(attn_output)
#         out = smirk_features_n + attn_output
#         out = self.fc(out)
        
#         return out


# class SmirkNet(nn.Module):
#     def __init__(self, num_class=8, num_head=1, dropout_rate=0.1):
#         super(SmirkNet, self).__init__()
#         self.bn1 = nn.BatchNorm1d(358)
        
#         self.self_attention = nn.MultiheadAttention(embed_dim=1, num_heads=num_head, dropout=dropout_rate, batch_first=True)
    
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(358, num_class)
       
#     def forward(self, x, smirk_feature): 
#         device = x.device
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]
        
#         smirk_features = torch.cat(smirk_features_list, dim=1)  # torch.Size([batch_size, 358])
        
#         # Apply batch normalization
#         smirk_features_n = self.bn1(smirk_features) # torch.Size([256, 358])
#         smirk_features = smirk_features_n.unsqueeze(2) # torch.Size([256, 358, 1])
#         #pdb.set_trace()
#         # Apply self-attention
#         attn_output, _ = self.self_attention(smirk_features, smirk_features, smirk_features) # torch.Size([256, 358, 1]) #query, key, value
#         attn_output = attn_output.squeeze(2) # torch.Size([256, 358])
#         attn_output = self.dropout(attn_output)
#         out = smirk_features_n + attn_output
#         out = self.fc(out)
        
#         return out

# class SmirkNet(nn.Module):
#     def __init__(self, num_class=8, num_head=2, embed_dim=128, dropout_rate=0.1):
#         super(SmirkNet, self).__init__()
#         self.bn1 = nn.BatchNorm1d(55)
#         self.embed_dim=embed_dim
#         # 线性层，用于将每个元素转换为高维嵌入
#         self.element_embedding = nn.Linear(1, embed_dim)
        
#         # 自注意力层
#         self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_head, dropout=dropout_rate, batch_first=True)
        
#         # 前馈神经网络
#         self.feed_forward = nn.Sequential(
#             nn.Linear(embed_dim, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, embed_dim),
#         )
        
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layernorm1 = nn.LayerNorm(embed_dim)
#         self.layernorm2 = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, num_class)
       
#     def forward(self, x, smirk_feature): 
#         device = x.device

#         # 将输入转为列表
#         smirk_features_list = [
#             torch.stack([f.clone().detach().to(device).float() for f in smirk_feature[key]], dim=1) for key in smirk_feature
#         ]
      
#         smirk_features = torch.cat(smirk_features_list, dim=1)  # torch.Size([batch_size, 55])

#         # 批量归一化
#         smirk_features = self.bn1(smirk_features)  # torch.Size([batch_size, 55])
        
#         # 将输入 reshape 为 [batch_size * 55, 1]
#         smirk_features = smirk_features.view(-1, 1) # torch.Size([batch_size * 55, 1]) # torch.Size([14080, 1])
        
#         # 线性层将每个元素转换为高维嵌入
#         smirk_features = self.element_embedding(smirk_features)  # torch.Size([batch_size * 55, embed_dim])
        
#         # 重塑为 [batch_size, 55, embed_dim]
#         smirk_features = smirk_features.view(-1, 55, self.embed_dim)  # torch.Size([batch_size, 55, embed_dim])
#         # 自注意力机制
#         attn_output, _ = self.self_attention(query=smirk_features,key= smirk_features, value=smirk_features)
#         attn_output = self.dropout(attn_output)
        
#         # 残差连接和层归一化
#         out1 = self.layernorm1(smirk_features + attn_output)
        
#         # 前馈神经网络
#         ff_output = self.feed_forward(out1)
#         ff_output = self.dropout(ff_output)
        
#         # 残差连接和层归一化
#         out2 = self.layernorm2(out1 + ff_output)
        
#         # 全连接层输出分类
#         out = self.fc(out2.mean(dim=1))  # 对维度进行平均以适应全连接层输入
#         # out = self.fc(out2)

#         return out

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
if __name__ == '__main__':
    smirk_feature = {
    "key1": [torch.randn(358) for _ in range(32)],
    "key2": [torch.randn(358) for _ in range(32)]
}
    model = SmirkNet()
    x = torch.randn(32, 100)  # Dummy input for x
    out = model(x, smirk_feature)