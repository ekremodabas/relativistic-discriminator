from models import DCGAN_64_Discriminator, DCGAN_64_Generator, StandartCNN_Discriminator, StandartCNN_Generator
from torch.utils.data import Dataset as dst
from glob import glob
import torch
import torch.nn as nn
from torch.cuda import FloatTensor as Tensor
from torch import clamp
from torch.autograd import grad
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import os
import sys
from PIL import Image
 
class Dataset(dst):

    def __init__(self, root, transforms):

        self.files = sorted(glob(root + '/*.png')) + sorted(glob(root + '/*.jpg'))

        self.transforms = transforms

    def __getitem__(self,index):

        return self.transforms(Image.open(self.files[index]))

    def __len__(self):

        return len(self.files)



def get_model(args):

    if(args.model == "standart_cnn"):

        return StandartCNN_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh) , StandartCNN_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm)

    if(args.model == "dcgan_64"):

        return DCGAN_64_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh) , DCGAN_64_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm)


def get_loss(model_type):

    if(model_type == "sgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss(C_real,ones) + loss(C_fake,zeros))

        def gen_loss(C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return loss(C_fake,ones)

        return gen_loss, disc_loss

    elif(model_type == "rsgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return loss((C_real-C_fake),ones)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return loss((C_fake-C_real),ones)
        

        return gen_loss, disc_loss

    elif(model_type == "rasgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 

        return gen_loss, disc_loss

    elif(model_type == "lsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss(C_real, zeros) + loss(C_fake, ones))

        def gen_loss(C_fake):
            
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return loss(C_fake,zeros)

        return gen_loss, disc_loss

    elif(model_type == "ralsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real-C_avg_fake), ones) + loss((C_fake-C_avg_real), -ones))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_fake-C_avg_real), ones) + loss((C_real-C_avg_fake),-ones)) 

        return gen_loss, disc_loss

    elif(model_type == "hingegan"):

        def disc_loss(C_real, C_fake):

            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (clamp((ones-C_real), min=0).mean() + clamp((C_fake+ones), min=0).mean())

        def gen_loss(C_fake):
            
            return -C_fake.mean()

        return gen_loss, disc_loss

    elif(model_type == "rahingegan"):

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (clamp((ones - C_real + C_avg_fake), min=0).mean() + clamp((ones + C_fake-C_avg_real), min=0).mean())

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (clamp((ones - C_fake + C_avg_real), min=0).mean() + clamp((ones + C_real - C_avg_fake), min=0).mean())

        return gen_loss, disc_loss

    elif(model_type == "wgan-gp"):

        def disc_loss(C_real, C_fake, grad_pen):
            
            return (-C_real.mean() + C_fake.mean() + grad_pen)

        def gen_loss(C_fake):
            
            return -C_fake.mean()

        return gen_loss, disc_loss

    elif(model_type == "rsgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, grad_pen):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real - C_fake),ones) + grad_pen)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            return loss((C_fake-C_real), ones)

        return gen_loss, disc_loss

    elif(model_type == "rasgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake, grad_pen):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros) + grad_pen)

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones(size=C_fake.size(), device=torch.device('cuda:0'))
            zeros = torch.zeros(size=C_fake.size(), device=torch.device('cuda:0'))
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 

        return gen_loss, disc_loss


def grad_penalty(Critic, x_hat, Lambda):

    x_hat.requires_grad_(True)
    critic_out = Critic(x_hat)
    grads = grad(outputs=critic_out, inputs=x_hat,
                 grad_outputs = torch.ones(critic_out.size()).cuda(),
                 create_graph=True)[0].view(x_hat.size(0),-1)

    return Lambda * torch.mean((grads.norm(p=2, dim=1) - 1)**2)



def get_dataset(dataset):

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    if(dataset == "cifar10"):

        return CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

    if(dataset == "cat"):

        return Dataset(root=os.path.join("datasets", "cats_64"), transforms=transform)
    
def cycle(iterable, dset):
    while True:
        for x in iterable:
            yield (x[0] if dset=='cifar10' else x)
            
def sampl(noise, Gen, iter):
    
    with torch.no_grad():
        
        out = Gen(noise)
        
        fig=plt.figure(figsize=(5,3))
        columns = 5
        rows = 2
        for i in range(1, 6):
            img = (out[i-1].permute(1,2,0).cpu()+1)*0.5
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        for i in range(6, 11):
            img = (out[i-1].permute(1,2,0).cpu()+1)*0.5
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show(block=False)
#         fig.savefig(f"im/{iter}.png", dpi=200)

def sample_fid(generator, it, args, step=500):
    
    #os.makedirs(f"fid/samplez/{args.dataset}_{args.model_type}_n_d_{args.d_iter}_b_size_{args.batch_size}_lr_{args.lr}_{it+1}")
    generator.eval()
    with torch.no_grad():
        for i in range(0,args.fid_sample, step):
            sys.stdout.write(f"\rsaving {i}/{args.fid_sample}")
            if(args.fid_sample < step+i):
                step = args.fid_sample-i
            geno = (generator(torch.randn(step,128,1,1).cuda())+1)*127.5 
            if(i == 0):
                arr = np.round_(geno.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)
            else:
                arr = np.concatenate((arr, np.round_(geno.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)), axis=0)
            
            #for j in range(step):
                #save_image(geno[j], f"fid/samplez/{args.dataset}_{args.model_type}_n_d_{args.d_iter}_b_size_{args.batch_size}_lr_{args.lr}_{it+1}/{i+j}.png")
        np.savez_compressed(f"fid/samplez/{args.dataset}_{args.model_type}_n_d_{args.d_iter}_b_size_{args.batch_size}_lr_{args.lr}_{it+1}", images=arr)
    generator.train()
        
        
        
