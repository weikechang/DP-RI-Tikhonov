import torch
import torch.nn as nn

#==================Unet utils====================
class conv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(conv, self).__init__()
        self.c = nn.Sequential(nn.Conv3d(inC, outC, kernel_size=kernel_size, padding=padding),
                               nn.ReLU(inplace=True))    
    def forward(self, x):
        x = self.c(x)
        return x

class convT(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding,output_padding, stride):
        super(convT, self).__init__()
        self.cT = nn.Sequential(nn.ConvTranspose3d(inC, outC, kernel_size=kernel_size, 
                                                   padding=padding,output_padding=output_padding,
                                                   stride=stride),
                                nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.cT(x)
        return x

class double_conv(nn.Module):
  #conv --> BN --> ReLUx2
    def __init__(self, inC, outC, kernel_size, padding):
        super(double_conv, self).__init__()
        self.conv2x = nn.Sequential(
            conv(inC, outC, kernel_size=kernel_size, padding=padding),
            conv(outC, outC, kernel_size=kernel_size, padding=padding))
   
    def forward(self, x):
        x = self.conv2x(x)
        return x

class down(nn.Module):
    def __init__(self, inC, outC):
        super(down, self).__init__()
        #go down = maxpool + double conv nn.MaxPool(2)             nn.Conv2d(inC, outC, kernel_size=2, stride=2, padding=0)
        self.go_down = nn.Sequential(
            nn.MaxPool3d(kernel_size = (2,2,2), stride = (2,2,2)),
            double_conv(inC, outC, kernel_size=3, padding=1))   
    def forward(self, x):
        x = self.go_down(x)
        return x

class up(nn.Module):
    def __init__(self, inC, outC):
        super(up, self).__init__()
        #go up = conv2d to half C-->upsample
        self.convt1 = convT(inC, inC, 3, stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.conv2x = double_conv(inC + outC, outC, kernel_size=3, padding=1)
    
    def forward(self, x1, x2):
        #x1 is data from a previous layer, x2 is current input
        x2 = self.convt1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2x(x)
        return x

class outconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_channels=inC, out_channels=outC, kernel_size=kernel_size, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x
    
#=================Unet=====================
class Unet(nn.Module):
    def __init__(self,C_size = 64):
        super(Unet, self).__init__()
        inC=1
        outC=1
        self.inc = double_conv(inC, C_size, 3, 1)
        self.down1 = down(C_size, C_size*2)
        self.down2 = down(C_size*2, C_size*4)
        self.down3 = down(C_size*4, C_size*8)

        self.up1 = up(C_size*8, C_size*4)
        self.up2 = up(C_size*4, C_size*2)
        self.up3 = up(C_size*2, C_size)
        self.outc = outconv(C_size, outC, 1, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x0 = self.up1(x3, x4)
        x0 = self.up2(x2, x0)
        x0 = self.up3(x1, x0)
        x0 = self.outc(x0)
        return x0 + x

if __name__ == '__main__':
    x = torch.rand(1,1,64,64,64)
    net = Unet()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.shape)