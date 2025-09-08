import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import numpy
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.parallel
import torch.backends.cudnn as cudnn
batch_size=64
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################
#   @author: Youngeun Kim and Priya Panda   #
#############################################
#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch.optim as optim
import torchvision
from   torch.utils.data.dataloader import DataLoader
from   torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os.path
import numpy as np
import torch.backends.cudnn as cudnn
from code_t.utills import *

cudnn.benchmark = True
cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
parser = argparse.ArgumentParser(description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=1003,        type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=20,    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=32,       type=int,   help='Batch size')
parser.add_argument('--lr',                    default=0.1,   type=float, help='Learning rate')
parser.add_argument('--leak_mem',              default=1.0,   type=float, help='Leak_mem')
parser.add_argument('--arch',              default='vgg9',   type=str, help='Dataset [vgg9, vgg11]')
parser.add_argument('--dataset',              default='cifar10',   type=str, help='Dataset [cifar10, cifar100]')
parser.add_argument('--num_epochs',            default=300,       type=int,   help='Number of epochs')
parser.add_argument('--num_workers',           default=0, type=int, help='number of workers')
parser.add_argument('--train_display_freq',    default=1, type=int, help='display_freq for train')
parser.add_argument('--test_display_freq',     default=1, type=int, help='display_freq for test')


global args
args = parser.parse_args()



#--------------------------------------------------
# Initialize tensorboard setting
#--------------------------------------------------
log_dir = 'modelsave'
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)


user_foldername = (args.dataset)+(args.arch)+'_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem)



#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
batch_size      = args.batch_size
batch_size_test = args.batch_size
num_epochs      = args.num_epochs
num_steps       = args.num_steps
lr   = args.lr


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    num_cls = 100
    img_size = 32

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
else:
    print("not implemented yet..")
    exit()



trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
#x, y = next(iter(trainloader))
#print(x.shape, y.shape,  x.min(), x.max())
#--------------------------------------------------
class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))

class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*1.0).float(),torch.sign(input))
		return out
#--------------------------------------------------
class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=1.0, img_size=32,default_threshold = 1.0,num_cls=10):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        #self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        #self.threshold=nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        #self.register_buffer('threshold', torch.tensor([1.]))
        #self.threshold = nn.ParameterDict()
        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine_flag = False
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv7 = nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool3 = nn.AvgPool2d(kernel_size=4)


        self.fc1 = nn.Linear((self.img_size//4)*(self.img_size//4)*256, 512, bias=bias_flag)
        self.bntt_fc = nn.BatchNorm1d(512, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.fc2 = nn.Linear(512, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]
        self.fc_list=[self.fc1,self.fc2]
        self.threshold1=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem1=nn.Parameter(torch.tensor(leak_mem))
        #print("self.leak_mem1",self.leak_mem1.size())
        self.threshold2=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem2=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold3=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem3=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold4=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem4=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold5=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem5=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold6=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem6=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold7=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem7=nn.Parameter(torch.tensor(leak_mem))
        
        self.threshold8=nn.Parameter(torch.tensor(default_threshold))
        self.leak_mem8=nn.Parameter(torch.tensor(leak_mem))
        
        # Turn off bias of BNTT
        #for bn_list in self.bntt_list:
            #for bn_temp in bn_list:
                #bn_temp.bias = None

        
        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                #m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                #m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)


    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv4 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv5 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv6 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size, self.img_size).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 512).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp
            mem_conv1 =  mem_conv1 + self.bntt1(self.conv1(out_prev))
            mem_thr = (mem_conv1 / self.threshold1 - 1.0)
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr > 0] = 1.0			
            mem_conv1 = mem_conv1-mem_conv1*rst
            out_prev1 = out.clone()

            mem_conv2 =  mem_conv2 + self.bntt2(self.conv2(out_prev1))			
            mem_thr = (mem_conv2 / self.threshold2) - 1.0  
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr > 0] = 1.0		
            mem_conv2 = mem_conv2-mem_conv2*rst
            out_prev2 = out.clone()  

            mem_conv3 =  mem_conv3 + self.bntt3(self.conv3(out_prev2)+out_prev1)
            mem_thr = (mem_conv3 / self.threshold3) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = 1.0			
            mem_conv3 = mem_conv3-mem_conv3*rst
            out_prev3 = out.clone()
           
        
            mem_conv4 =  mem_conv4 + self.bntt4(self.conv4(out_prev3) +out_prev2)
            mem_thr = (mem_conv4 / self.threshold4) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = 1.0		
            mem_conv4 = mem_conv4-mem_conv4*rst
            out_prev4 = out.clone()
           
 
            mem_conv5 =  mem_conv5 + self.bntt5(self.conv5(out_prev4)+out_prev3)
            mem_thr = (mem_conv5 / self.threshold5) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = 1.0		
            mem_conv5 = mem_conv5-mem_conv5*rst
            out_prev5 = out.clone()

            mem_conv6 =  mem_conv6 + self.bntt6(self.conv6(out_prev5)+out_prev4)
            mem_thr = (mem_conv6 / self.threshold6) - 1.0
            out = self.spike_fn(mem_thr)    
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = 1.0		
            mem_conv6 = mem_conv6-mem_conv6*rst
            out_prev6 = out.clone()
     
            mem_conv7 =  mem_conv7 + self.bntt7(self.conv7(out_prev6))
            mem_thr = (mem_conv7 / self.threshold7) - 1.0
            out = self.spike_fn(mem_thr)    
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = 1.0		
            mem_conv7 = mem_conv7-mem_conv7*rst
            out_prev8 = out.clone()
            
            out_prev8=self.pool3(out_prev8)
            out_prev8 = out_prev8.reshape(batch_size, -1)

            mem_fc1 = mem_fc1 + self.bntt_fc(self.fc1(out_prev8)) 
            mem_thr = (mem_fc1 / self.threshold8) - 1.0  ###
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = 1.0			
            mem_fc1 = mem_fc1-mem_fc1*rst
            out_prev9 = out.clone()
   
            mem_fc2 = mem_fc2 + self.fc2(out_prev9)
       

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage
#--------------------------------------------------
class SNN_VGG11_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=100):
        super(SNN_VGG11_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.input_layer 	= PoissonGenerator()
        print (">>>>>>>>>>>>>>>>> VGG11 >>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = False
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        
        self.shortcut2 = nn.Conv2d(32, 256, kernel_size=1, stride=2, bias=False)
        self.s2=nn.BatchNorm2d(256)

        
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt9 = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag)
        
        self.pool4 = nn.AvgPool2d(kernel_size=4)

        self.fc1 = nn.Linear((self.img_size//4)*(self.img_size//4)*512, 1024, bias=bias_flag)
        self.bntt_fc = nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) 
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)
        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt8, self.bntt_fc]
        self.pool_list = [self.pool1, self.pool2, False, self.pool3, False, self.pool4, False]

        # Turn off bias of BNTT
        #for bn_list in self.bntt_list:
            #for bn_temp in bn_list:
                #bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv5 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv6 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size, self.img_size).cuda()
        mem_conv8 = torch.zeros(batch_size, 256, self.img_size//2, self.img_size//2).cuda()
        mem_conv9 = torch.zeros(batch_size, 512, self.img_size//2, self.img_size//2).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7,mem_conv8,mem_conv9]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp
            out_prev1 = self.bntt1(self.conv1(out_prev.float()))
            mem_conv1 = self.leak_mem * mem_conv1 + out_prev1
            mem_thr = (mem_conv1 / self.conv1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv1).cuda()
            rst[mem_thr > 0] = self.conv1.threshold
            mem_conv1 = mem_conv1 - rst
            out_prev1 = out.clone()
            
            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2(self.conv2(out_prev1))
            mem_thr = (mem_conv2 / self.conv2.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv2).cuda()
            rst[mem_thr > 0] = self.conv2.threshold
            mem_conv2 = mem_conv2 - rst
            out_prev2 = out.clone()
            
            mem_conv3 = self.leak_mem * mem_conv3 + self.bntt3(self.conv3(out_prev2)+out_prev1)
            mem_thr = (mem_conv3 / self.conv3.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst[mem_thr > 0] = self.conv3.threshold
            mem_conv3 = mem_conv3 - rst
            out_prev3 = out.clone()

            
            mem_conv4 = self.leak_mem * mem_conv4 + self.bntt4(self.conv4(out_prev3)+out_prev2)
            mem_thr = (mem_conv4 / self.conv4.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst[mem_thr > 0] = self.conv4.threshold
            mem_conv4 = mem_conv4 - rst
            out_prev4 = out.clone()
			
            mem_conv5 = self.leak_mem * mem_conv5 + self.bntt5(self.conv5(out_prev4)+out_prev3)
            mem_thr = (mem_conv5 / self.conv5.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst[mem_thr > 0] = self.conv5.threshold
            mem_conv5 = mem_conv5 - rst
            out_prev5 = out.clone()

            mem_conv6 = self.leak_mem * mem_conv6 + self.bntt6(self.conv6(out_prev5)+out_prev4)
            mem_thr = (mem_conv6 / self.conv6.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.conv6.threshold
            mem_conv6 = mem_conv6 - rst
            out_prev6 = out.clone()

            mem_conv7 = self.leak_mem * mem_conv7 + self.bntt7(self.conv7(out_prev6)+out_prev5)
            mem_thr = (mem_conv7 / self.conv7.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = self.conv7.threshold
            mem_conv7 = mem_conv7 - rst
            out_prev7 = out.clone()

            
            mem_conv8 = self.leak_mem * mem_conv8 + self.bntt8(self.conv8(out_prev7))
            mem_thr = (mem_conv8 / self.conv8.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv8).cuda()
            rst[mem_thr > 0] = self.conv8.threshold
            mem_conv8 = mem_conv8 - rst
            out_prev8 = out.clone()
            
            mem_conv9 = self.leak_mem * mem_conv9 + self.bntt9(self.conv9(out_prev8))
            mem_thr = (mem_conv9 / self.conv9.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv9).cuda()
            rst[mem_thr > 0] = self.conv9.threshold
            mem_conv9 = mem_conv9 - rst
            out_prev9 = out.clone()
            out_prev=self.pool3(out_prev9)
            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc(self.fc1(out_prev))  ### the last layer input
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0  ###
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage
#--------------------------------------------------
# Instantiate the SNN model and optimizer
#--------------------------------------------------

model =  SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
model = model.cuda()
#--------------------------------------------------

#--------------------------------------------------

class BinOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

bin_op = BinOp(model)
##################################################################

# weight_list, bias_list = get_weight(model)
# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
print('********** SNN simulation parameters **********')
print('Simulation # time-step : {}'.format(num_steps))
print('Membrane decay rate : {0:.2f}\n'.format(leak_mem))

print('********** SNN learning parameters **********')
print('Backprop optimizer     : SGD')
print('Batch size (training)  : {}'.format(batch_size))
print('Batch size (testing)   : {}'.format(batch_size_test))
print('Number of epochs       : {}'.format(num_epochs))
print('Learning rate          : {}'.format(lr))


#--------------------------------------------------
#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
print('********** SNN training and evaluation **********')
train_loss_list = []
test_acc_list = []
for epoch in range(num_epochs):
    train_loss = AverageMeter()
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        bin_op.binarization()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = model(inputs)
        
        loss   = criterion(output, labels)

        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        train_loss.update(loss.item(), labels.size(0))

        loss.backward()
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()

    if (epoch+1) % args.train_display_freq ==0:
        print("Epoch: {}/{};".format(epoch+1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

    adjust_learning_rate(optimizer, epoch, num_epochs)



    if (epoch+1) %  args.test_display_freq ==0:
        acc_top1, acc_top5 = [], []
        model.eval()
        bin_op.binarization()
        with torch.no_grad():
             
             for j, data in enumerate(testloader, 0):
                 images, labels = data
                 images = images.cuda()
                 labels = labels.cuda()
                 #print(labels)

                 out = model(images)
                 prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                 acc_top1.append(float(prec1))
                 acc_top5.append(float(prec5))


        test_accuracy = np.mean(acc_top1)
        print ("test_accuracy : {}". format(test_accuracy))


#print("##############################################################")
sys.exit(0)