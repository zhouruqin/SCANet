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
        values, index = x.topk(self.k, self.dim, largest=True, sorted=True)
        return values, index 


# basic conv block
class BasicConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active = True):
        super(BasicConv1D, self).__init__()
        self. active = active
        #self.bn = nn.BatchNorm1d(in_channels)
        if self.active == True:
            self.activation = Mish()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False)
        #self.dropout = nn.Dropout(0.5) 
       

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

#basic self-attention block
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, out_dim):
        super(Self_Attn,self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        
        #查询卷积
        self.query_conv =BasicConv1D(in_dim, out_dim)  
        '''nn.Sequential(
             Resblock1D(in_dim, out_dim),
             Resblock1D(out_dim, out_dim)
             )# '''
       
            


        self.value_conv = nn.Sequential(
             Resblock1D(in_dim, out_dim),
             Resblock1D(out_dim, out_dim)
             )


        #if in_dim != out_dim:
        #    self.short_conv = BasicConv1D(in_dim, out_dim)

        #self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N)  32, 1024, 64
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """

        proj_query  = self.query_conv(x).permute(0,2,1)      # B, in_dim, N   ---> B, in_dim // 8, N   ---->  B, N, in_dim // 8
        proj_key =   proj_query.permute(0,2,1) #B, in_dim, N   ---> B, in_dim // 8, N
        
        energy =  torch.bmm(proj_query,proj_key) # transpose check    B, N, N

        attention = self.softmax(energy) # B , N,  N  
        
        proj_value = self.value_conv(x)  #proj_key# #B, in_dim, N ----> B, out_dim, N
        #proj_value = self.value_conv_1(x)
        #proj_value = self.value_conv_2(proj_value)

        #attention = torch.where(attention>1/self.out_dim, attention, torch.full_like(attention, 0))#
        
        out_x = torch.bmm(proj_value, attention.permute(0,2,1) )   #B, out_dim, N
        
        #if self.in_dim != self.out_dim:
        #    x = self.short_conv(x)#self.gamma*+ proj_valuesself.beta *
         
        out =  self.beta * out_x + proj_value #+ proj_value #self.short_conv(x)self.alpha* 
        #out = torch.cat([proj_value, out_x], dim =1)
        #out_K = out.topk(int(out.size()[1]/2),1)
        
        return out#, out1


#------------------------------------------------------------------#
#pointnet
class PointNetFeatures(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape

        self.conv1 = BasicConv1D(3, 64)# 
        self.conv2 = BasicConv1D(64, 64)#
        self.conv3 = BasicConv1D(64, 64)#
        self.conv4 = BasicConv1D(64, 128)#
        self.conv5 = BasicConv1D(128, bottleneck_size)#

    def forward(self, x):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y =(self.conv1(x))# 
        y = (self.conv2(y))#
        y = (self.conv3(y))#
        y = (self.conv4(y))#
        y = (self.conv5(y))  # Batch x 1024 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 1024
        y = y.contiguous()

        return y

#SSA
class PointNetSAFeatures(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bnc"):
        super(PointNetSAFeatures, self).__init__()

        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
 
        self.conv0 = BasicConv1D(3, 64)#

        self.kmax_pooling4 = kmax_pooling(2, 4)
        self.Self_Attn1 = Self_Attn(64, 64)
        self.Self_Attn2 = Self_Attn(64, 64)  #
        self.Self_Attn3 = Self_Attn(64, 128) #
       
    
    def forward(self, x):#
        if self.input_shape == "bnc":   #换成BCN
            x = x.permute(0, 2, 1)
        
        if x.shape[1] != 3 :#
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")     
        
 
        x0 = self.conv0(x )#64
        
        x1 = self.Self_Attn1(x0)# 64
    
        x2 = self.Self_Attn2(x1+x0)# 64
       
        x3 = self.Self_Attn3(x2+x1+x0)# 128

        x5  = torch.cat([ x1, x2,x3], dim=1)  #256
        x, _ = self.kmax_pooling4 (x5)  #3
        x = x.view(x.size()[0], -1)

        x = x.contiguous()
        
        return x
 
