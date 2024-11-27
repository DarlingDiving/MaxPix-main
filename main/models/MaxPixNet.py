
import torch
import torch.nn as nn
from .basicmodule import BasicModule
import models.MResNet as MResNet
import torch.nn.functional as F

MResNet = MResNet.resnet18_MABlock(pretrained=False)


class MaxSel(nn.Module):
    def __init__(self):
        super(MaxSel, self).__init__()

        kernel_1 = [[0,0,0],[1,-2,1],[0,0,0]]
        kernel_2 = [[0,1,0],[0,-2,0],[0,1,0]]
        kernel_3 = [[0,0,1],[0,-2,0],[1,0,0]]
        kernel_4 = [[1,0,0],[0,-2,0],[0,0,1]]

        kernel_1 = torch.FloatTensor(kernel_1).unsqueeze(0).unsqueeze(0)
        self.weight_1 = nn.Parameter(data=kernel_1, requires_grad=False)

        kernel_2 = torch.FloatTensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight_2 = nn.Parameter(data=kernel_2, requires_grad=False)

        kernel_3 = torch.FloatTensor(kernel_3).unsqueeze(0).unsqueeze(0)
        self.weight_3 = nn.Parameter(data=kernel_3, requires_grad=False)

        kernel_4 = torch.FloatTensor(kernel_4).unsqueeze(0).unsqueeze(0)
        self.weight_4 = nn.Parameter(data=kernel_4, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1_1 = F.conv2d(x1.unsqueeze(1), self.weight_1, padding=1)
        x2_1 = F.conv2d(x2.unsqueeze(1), self.weight_1, padding=1)
        x3_1 = F.conv2d(x3.unsqueeze(1), self.weight_1, padding=1)

        x1_2 = F.conv2d(x1.unsqueeze(1), self.weight_2, padding=1)
        x2_2 = F.conv2d(x2.unsqueeze(1), self.weight_2, padding=1)
        x3_2 = F.conv2d(x3.unsqueeze(1), self.weight_2, padding=1)

        x1_3 = F.conv2d(x1.unsqueeze(1), self.weight_3, padding=1)
        x2_3 = F.conv2d(x2.unsqueeze(1), self.weight_3, padding=1)
        x3_3 = F.conv2d(x3.unsqueeze(1), self.weight_3, padding=1)

        x1_4 = F.conv2d(x1.unsqueeze(1), self.weight_4, padding=1)
        x2_4 = F.conv2d(x2.unsqueeze(1), self.weight_4, padding=1)
        x3_4 = F.conv2d(x3.unsqueeze(1), self.weight_4, padding=1)

        x_c1 = torch.cat((x1_1,x1_2,x1_3,x1_4), dim=1)
        x_c2 = torch.cat((x2_1,x2_2,x2_3,x2_4), dim=1)
        x_c3 = torch.cat((x3_1,x3_2,x3_3,x3_4), dim=1)
        x_1,_ = torch.max(x_c1,dim=1)
        x_1 = x_1.unsqueeze(0).transpose(0,1)

        x_2,_ = torch.max(x_c2,dim=1)
        x_2 = x_2.unsqueeze(0).transpose(0,1)

        x_3,_ = torch.max(x_c3,dim=1)
        x_3 = x_3.unsqueeze(0).transpose(0,1)
        
        x = torch.cat((x_1,x_2,x_3), dim=1)
       
        return x

class MaxPix(BasicModule):
    def __init__(self, num_classes=None):
        super(MaxPix, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(8192, num_classes)
        self.MResNet = MResNet
        self.MaxSel = MaxSel()
            
    def forward(self, rgb_data):
        rgb_data = self.MaxSel(rgb_data)
        output = self.MResNet(rgb_data)   

        # decision
        final_out1 = torch.flatten(output, 1)
        final_out = self.fc(final_out1)
        return final_out,final_out1

