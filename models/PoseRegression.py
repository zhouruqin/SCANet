from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------#
#Mish
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

#k pool
class kmax_pooling(nn.Module):
    def __init__(self, dim, k):
        super(kmax_pooling, self).__init__()
        self.k =k
        self.dim = dim

    def forward(self, x):
        #index = x.topk(self.k, dim=self.dim)[1]#.sort(dim=self.dim)[0]
        values, index = x.topk(self.k, self.dim, largest=True, sorted=True)
        return values, index #x.gather(self.dim, index), index


#basic conv block
class BasicConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active = True):
        super(BasicConv1D, self).__init__()
        self. active = active
        #self.bn = nn.BatchNorm1d(in_channels)
        if self.active == True:
            self.activation = Mish()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False)

    def forward(self, x):
        #x = self.bn(x)
        if self.active == True:
            x = self.activation(x)
       
        x = self.conv(x)
       
        return x

#basic residual block
class Resblock1D(nn.Module):
    def __init__(self, channels, out_channels, residual_activation=nn.Identity()):
        super(Resblock1D, self).__init__()

        self.channels = channels
        self.out_channels = out_channels
        if self.channels!= self.out_channels:
            self.res_conv = BasicConv1D(channels, out_channels, 1)
       
        self.activation = Mish()
        self.block = nn.Sequential(
            BasicConv1D(channels, out_channels//2, 1)   ,
            BasicConv1D(out_channels//2, out_channels, 1 , active = False)       
        )

    def forward(self, x):
        residual =x 
        if self.channels!= self.out_channels:
            residual = self.res_conv(x)
        return self.activation(residual+self.block(x)) 



#cross 
class self_attention_fc (nn.Module):
    """ Self attention Layer""" 
    def __init__(self,in_dim, out_dim):     #1024
        super(self_attention_fc,self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.query_conv = BasicConv1D(in_dim, out_dim)

        #键值卷积
        #self.key_conv = BasicConv1D(in_dim, out_dim)
        
        #self.value_conv_x = BasicConv1D(in_dim, out_dim)

        #self.value_conv_y = BasicConv1D(in_dim, out_dim)
        

        #self.kama = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
        #self.select_k = select_k()

    def forward(self,x, y):   #B, 1024 , 1
        """
            inputs :
                x : input feature maps( B X C,1 )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        proj_query_x  = self.query_conv(x)   #[B, in_dim, 1]----->[B, out_dim1, 1]

        proj_key_y =   self.query_conv(y).permute(0,2,1)          #[B, 1, out_dim1]
        
        #energy_x = torch.bmm(proj_query_x,proj_key_x)
        #energy_y = torch.bmm(proj_query_y,proj_key_y)
        energy_xy =  torch.bmm(proj_query_x,proj_key_y) #  xi 对 y所有点的注意力得分  [B, 64, 64]   
       
        #attention_x = self.softmax(energy_x) #  按行归一化  xi 对 y所有点的注意力
        #attention_y = self.softmax(energy_y)
        attention_xy = self.softmax(energy_xy)
        attention_yx = self.softmax(energy_xy.permute(0,2,1))
     
        proj_value_x = proj_query_x#self.value_conv_x(x) # [B, out_dim, 64]
        proj_value_y = proj_key_y.permute(0,2,1)  #self.value_conv_x(y) # [B, out_dim, 64]

        
        #value_x, index_x = self.select_k(attention_xy)
        #out_x = proj_value_x.squeeze(2).gather(1, index_x)


        #value_y, index_y = self.select_k(attention_yx)
        #out_y = proj_value_y.squeeze(2).gather(1, index_y)
        out_x = torch.bmm(attention_xy, proj_value_x) #  [B, out_dim]
        out_x =  self.beta* out_x +  proj_value_x #self.kama* 

        out_y = torch.bmm(attention_yx, proj_value_y ) #  [B, out_dim]
        out_y =  self.beta*out_y +   proj_value_y  #self.kama *
 
        return out_x,out_y

#--------------------------------------------------------------#
class pose_regression_fc(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        
        self.fc1 = nn.Linear(bottleneck_size * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)   #1024, 1024
        self.fc3 = nn.Linear(1024, 512) #1024, 512
        self.fc4 = nn.Linear(512, 512)#512, 512
        self.fc5 = nn.Linear(512, 256)#512, 256
        self.fc6 = nn.Linear(256, 7)#256, 7

    def forward(self, x):
        
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))
        y = F.relu(self.fc5(y))
        y = self.fc6(y)  # Batch x 7

        pre_normalized_quat =y[:, 0:4]
        normalized_quat = F.normalize(pre_normalized_quat, dim=1)
        trans = y[:, 4:]
        y = torch.cat([normalized_quat, trans], dim=1)

        return y


class pose_regression_CMA(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        
        self.fc1 = self_attention_fc(bottleneck_size, 512)# 
        self.fc1_ = nn.Sequential(Mish(),nn.Linear(1024, 512) )#
       
        self.fc2 = self_attention_fc(512, 256)#
        self.fc2_ = nn.Sequential( Mish(), nn.Linear(512, 256))#
    
        self.fc3 = self_attention_fc(256, 128)#
        self.fc3_ = nn.Sequential( Mish(), nn.Linear(256, 128))# 
      
        self.fc4 = self_attention_fc(128, 64)#
        self.fc4_ = nn.Sequential( Mish(), nn.Linear(128, 64))#

        self.fc6 = nn.Linear(64,7)#


    def forward(self, x, y): #B, 1024
        
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)


        x0, y0 = self.fc1(x, y)    #512
        delt_xy_0 = self.fc1_(torch.cat([x0.squeeze(2), y0.squeeze(2)], dim=1))
       
        x1, y1= self.fc2( x0, y0)    #256
        delt_xy_1 = self.fc2_(torch.cat([x1.squeeze(2), y1.squeeze(2)], dim=1)+delt_xy_0 )
      
        x2, y2=self.fc3(x1, y1)    #128
        delt_xy_2 = self.fc3_(torch.cat([x2.squeeze(2), y2.squeeze(2)], dim=1)+delt_xy_1)
       
        x3, y3= self.fc4(x2, y2)    #64
        delt_xy_3 = self.fc4_(torch.cat([x3.squeeze(2), y3.squeeze(2)], dim=1)+delt_xy_2 )
       
        x6 = self.fc6( delt_xy_3)


        pre_normalized_quat =x6[:, 0:4]
        normalized_quat = F.normalize(pre_normalized_quat, dim=1)
        trans = x6[:, 4:]
      
        x7 = torch.cat([normalized_quat, trans], dim=1)

        return   x7,x7

