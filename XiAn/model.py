import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
 
import numpy as np
import math
from torch.nn import BatchNorm2d, BatchNorm1d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d
import util

from utils import ST_BLOCK_0 #ASTGCN
from utils import ST_BLOCK_1 #DGCN_Mask/DGCN_Res
from utils import ST_BLOCK_2_r #DGCN_recent
from utils import ST_BLOCK_4 #Gated-STGCN
from utils import ST_BLOCK_5 #GRCN
from utils import ST_BLOCK_6 #OTSGGCN
from utils import multi_gcn #gwnet

from utils import Cross_M
from utils import Cross_P
"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class ASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,d_adj,t_adj

class PASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes,  masks,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(PASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_P(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,dist,poi
    
class MMASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes,  masks,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MMASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_M(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,dist,poi


class LSTM(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        
        
    def forward(self,input):
        x=input
        shape = x.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        c = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=(h,c)
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.lstm(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]

class GRU(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRU,self).__init__()
        self.gru=nn.GRU(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1,tem_size),
                          stride=(1,1), bias=True)
        
    def forward(self,input):
        x=input
        shape = x.shape
        h =Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=h
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.gru(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]
        
class Gated_STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(Gated_STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x,adj,adj 

class PSTGCN(nn.Module):
    def __init__(self,device, num_nodes, masks,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(PSTGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.ones(num_nodes,num_nodes), requires_grad=True)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_P(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x=self.conv1(x)#b,12,n,1
        return x,dist,poi  
    
    
class MMSTGCN(nn.Module):
    def __init__(self,device, num_nodes, masks,dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MMSTGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.ones(num_nodes,num_nodes), requires_grad=True)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_M(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x=self.conv1(x)#b,12,n,1
        return x,dist,poi    
    
    

class GRCN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRCN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_5(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        
        adj=self.supports[0]
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.conv1(x)
        return x,adj,adj     
#OGCRNN
class OGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5,self.training)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x,A,A   

class POGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, masks, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(POGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_P(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5,self.training)
        x=self.block1(x,A)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x=self.conv1(x)
        return x,dist,poi  
    
class MMOGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, masks, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MMOGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]

        self.DATT1=Cross_M(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5,self.training)
        x=self.block1(x,A)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x=self.conv1(x)
        return x,dist,poi    


#OTSGGCN    
class OTSGGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OTSGGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.block1=ST_BLOCK_6(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.ones(num_nodes,num_nodes), requires_grad=True)
        #nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=input#self.bn(input)
        mask=(self.supports[0]!=0).float()
        A=self.h*mask
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.3,self.training)
        A1=torch.eye(self.num_nodes).cuda()-A
       # A1=F.dropout(A1,0.5)
        x=self.block1(x,A1)
        x=self.block2(x,A1)
        x=self.block3(x,A1)
        x=self.conv1(x)#b,12,n,1
        return x,A1,A1     

#DGCN    
class DGCN(nn.Module):  
    def __init__(self,device, num_nodes, dropout,masks, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(DGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.block1=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.end_conv_1 = nn.Conv2d(in_channels=dilation_channels,
                                  out_channels=end_channels//4,
                                  kernel_size=(1,12),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels//4,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1))+0.00001
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
        
        x = self.start_conv(x)
        skip=0
        x,_,_=self.block1(x,A1)
        #print(x.shape)
        x,_,_=self.block2(x,A1)
        #print(x.shape)
        x = self.conv1(x)
        return x,A1,A1  

class PDGCN(nn.Module):  
    def __init__(self,device, num_nodes, dropout,masks, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(PDGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.block1=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.end_conv_1 = nn.Conv2d(in_channels=dilation_channels,
                                  out_channels=end_channels//4,
                                  kernel_size=(1,12),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels//4,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]
        self.DATT1=Cross_P(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1))+0.00001
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
        
        x = self.start_conv(x)
        skip=0
        x,_,_=self.block1(x,A1)
        #print(x.shape)
        x,_,_=self.block2(x,A1)
        #print(x.shape)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = self.conv1(x)
        return x,dist,poi     
    
class MMDGCN(nn.Module):  
    def __init__(self,device, num_nodes, dropout,masks, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(MMDGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.block1=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.end_conv_1 = nn.Conv2d(in_channels=dilation_channels,
                                  out_channels=end_channels//4,
                                  kernel_size=(1,12),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels//4,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]
        self.DATT1=Cross_M(dilation_channels,12,num_nodes)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1))+0.00001
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
        
        x = self.start_conv(x)
        skip=0
        x,_,_=self.block1(x,A1)
        #print(x.shape)
        x,_,_=self.block2(x,A1)
        #print(x.shape)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = self.conv1(x)
        return x,dist,poi          
    
#gwnet    
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)

    def forward(self, input):
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,adp,adp

class Pgwnet(nn.Module):
    def __init__(self, device, num_nodes, masks,dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(Pgwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]
        self.DATT1=Cross_P(skip_channels,1,num_nodes)

    def forward(self, input):
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,dist,poi


class MMgwnet(nn.Module):
    def __init__(self, device, num_nodes, masks,dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(MMgwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)
        self.mask1=masks[0]
        self.poi=masks[1]
        self.poi_vec=masks[2]
        self.DATT1=Cross_M(skip_channels,1,num_nodes)

    def forward(self, input):
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        dist,poi=self.DATT1(self.poi_vec,x,self.poi)
        x=x*(poi+1)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,dist,poi

    
