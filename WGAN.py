from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

workers=0 #number of data working loaders, for the data download
batchSize=64 #the number of mini batches passed at each batch pass
imageSize=64 #the height/width of the input image
nz=100 #size of latent vector z, i.e. the random numbers passed as inputs in the generator!
ngf=64 #number of generator filters in the first convolutional layer
ndf=64 #number of Discriminator filters in the first convolutional layer
niter=100 #number of training iterations/epochs
lrD=0.00005 #learning rate for Critic, default=0.00005
lrG=0.00005 #learning rate for Generator, default=0.00005
beta1=0.5 #beta1 for adam. default=0.5
ngpu=1 #number of gpu's to use
netG='' #path to netG (to continue training)
netD='' #path to netD (to continue training)
clamp_lower=-0.01
clamp_upper=0.01
Diters_init=5 #number of D iters per each G iter
n_extra_layers=0 #Number of extra layers on gen and disc
experiment='samples'
os.system('mkdir {0}'.format(experiment)) #create folder for pictures
manualSeed=random.randint(1,10000) #fix the seed used to illustrate the pictures during training
print ("Random seed: ", manualSeed)
random.seed=manualSeed
torch.manual_seed=manualSeed

cudnn.benchmark=True #uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                     #If this is set to false, uses some in-built heuristics that might not always be fastest.

cuda=True
if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.CIFAR10(root='cifar10', download=True,
                       transform=transforms.Compose([
                           transforms.Scale(imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
)

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=int(workers))

nc = 3 #number of color channels

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) #we initialize weights in the conv layer with a Gaussian with mean=0 and stddev=0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) #we initialize weights in the batchnorm layer with a Gaussian with mean=1 and stddev=0.02
        m.bias.data.fill_(0) #biases are initialized to zeros


# ----------------------------------- DCGAN for the Generator -----------------------------------------
class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4  # ngf=64 => cngf=32, tisize=4
        while tisize != isize:    # iter1 => cngf=64, tisize=8
            cngf = cngf * 2       # iter2 => cngf=128, tisize=16
            tisize = tisize * 2   # iter3 => cngf=256, tisize=32
                                  # iter4 => cngf=512, tisize=64
        main = nn.Sequential()
        # input is Z, going into a convolution
        # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
        #                                stride=1, padding=0, output_padding=0, groups=1, bias=True)
        
        # Input = (N, C_in, H_in, W_in)
        # Output = (N, C_out, H_out, W_out)
        # Here the height(H) and Width(W) of the output depends on the stride and padding, see:
        # http://pytorch.org/docs/nn.html
        # The PyTorch function, ConvTranspose2d() only takes the number of input and output channels
        # as arguments, the size of the image is deducted from the input passed at training.
        
        # NB: WE USE TRANSPOSED CONVOLUTIONS, NOT CONVOLUTIONS!
 
        # Transposed convolutions generally arises from the desire to use a transformation going 
        # in the opposite direction of a normal convolution, i.e., from something that has the shape 
        # of the output of some convolution to something that has the shape of its input while maintaining 
        # a connectivity pattern that is compatible with said convolution.
        
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        # class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        # Continue creating conv. layers such that we end up having 64 output channels, i.e.
        # the size of the picture
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))
        
        # add final layer to reduce the channels to nc, i.e. 3 color channels
        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main
        
        # An example of a structure given imageSize=isize=64, nz=100, nc=3, ngf=64, ngpu=1, n_extra_layers=0)
        # Conv. layer 1:
        # nn.ConvTranspose2D(in_channels=100, out_channels=504, kernel_size=4, stride=1, padding=0, bias=False)
        # Conv. layer 2:
        # nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)
        # cngf=256, csize=8
        # Conv. layer 3:
        # nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)
        # cngf=128, csize=16
        # Conv. layer 4:
        # nn.ConvTranspose2D(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        # cngf=64, csize=32
        # Conv. layer 5 (OUTPUT):
        # nn.ConvTranspose2D(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False)
        
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)


# Create a Generator
netG = DCGAN_G(imageSize, nz, nc, ngf, ngpu, n_extra_layers)

# initialize weights in Generator
netG.apply(weights_init)
print(netG) # Awesome stuff, my layer comment above verified!

# ----------------------------------- DCGAN for the Discriminator -----------------------------------------

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        #  class torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
        #                        stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        # Input = (N, C_in, H_in, W_in)
        # Output = (N, C_out, H_out, W_out)
        
        
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main
        
        
        # An example of structure given inputs: imageSize=64, nz=100, nc=3, ndf=64, ngpu=1, n_extra_layers=0):
        # Conv. Layer 1:
        # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False)
        # csize=32, cndf=64
        # Conv. Layer 2:
        # in_feat=64, out_feat=128
        # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False))
        # csize=16, cndf=128
        # Conv. Layer 3:
        # in_feat=128, out_feat=256
        # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False))
        # csize=8, cndf=256
        # Conv. Layer 4:
        # in_feat=256, out_feat=512
        # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False))
        # csize=4, cndf=512
        # Conv. Layer 5 (OUTPUT LAYER/FULLY CONNECTED LAYER):
        # in_feat=512, out_feat=1
        # nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        #print ("output of D before .mean(0) operation: ", output.size())
        output = output.mean(0)
        #print ("output of D after .mean(0) operation: ", output.size())
        return output.view(1)

# Create the Critic/Dicriminator
netD = DCGAN_D(imageSize, nz, nc, ndf, ngpu, n_extra_layers)

netD.apply(weights_init)
print(netD)

# This is just a check to see that an input image (32, 3, 64, 64) and a random vector (32, 100, 1, 1) achieves
# the correct output dimensions
m=Variable(torch.randn(32, 3, 64, 64))
m2=Variable(torch.randn(32, 100, 1, 1))
outputD=netD.forward(m)
outputG=netG.forward(m2)
outputG.size()

input = torch.FloatTensor(batchSize, nc, imageSize, imageSize) 
noise = torch.FloatTensor(batchSize, nz, 1, 1) 
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1]) 
mone = one * -1

if cuda:
    print("cuda available, optimizing variables for GPU")
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# Use RMSprop, for an explanation see p. 26-29 of 
# http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
optimizerD = optim.RMSprop(netD.parameters(), lr = lrD)
optimizerG = optim.RMSprop(netG.parameters(), lr = lrG)



# JUST TESTING STUFF; DELETE AFTER --------------------
print(len(dataloader))
"""
data_iter=iter(dataloader)

for p in netD.parameters():
    print(p.size())
    p.data.clamp_(clamp_lower, clamp_upper)
print(len(data_iter))

data=data_iter.next() # slow as fuck

data_iter=iter(dataloader)
for p in netD.parameters():
    p.data.clamp_(clamp_lower, clamp_upper)

data = data_iter.next()
print(data)
# train with real
real_cpu, _ = data
netD.zero_grad() # clear gradient, i.e. set to 0 before doing backward pass
batch_size = real_cpu.size(0) 

if cuda:
    real_cpu = real_cpu.cuda()
input.resize_as_(real_cpu).copy_(real_cpu)
inputv = Variable(input)

errD_real = netD(inputv)
errD_real.backward(one)

print(errD_real)
"""
# END OF TESTING ---------------------------------------

# ----------------------------- Run the algorithm ----------------------------------

gen_iterations = 0
for epoch in range(niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = Diters_init
        j = 0
        # if EITHER we have trained D for Diters=100 times or we have used all minibatches (len(dataloader))
        # we stop training D.
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            # clamping reduces all parameters to the range [clamp_lower ; clamp_upper]
            # This is not the same as normalization, as all weights that are already in the range
            # remain the same, while weights outside the range are clamped to either the lower or 
            # upper bound
            for p in netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad() # clear gradient, i.e. set to 0 before doing backward pass
            batch_size = real_cpu.size(0) 

            if cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv) # this corresponds to calling the forward function
            errD_real.backward(one)

            # train with fake
            noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            inputv = fake
            inputv.detach()
            errD_fake = netD(inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, niter, i, len(dataloader), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % 100 == 0:
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(experiment))
            fake = netG(Variable(fixed_noise, volatile=True))
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(experiment, gen_iterations))

    # do checkpointing
    #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(experiment, epoch))
    #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(experiment, epoch))




