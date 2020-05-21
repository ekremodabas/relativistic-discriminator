from models import DCGAN_64_Discriminator, DCGAN_64_Generator, StandardCNN_Discriminator, StandardCNN_Generator
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
from argparse import ArgumentTypeError
import tarfile
import urllib

class Dataset(dst):

    def __init__(self, root, transforms, download):

        if(not os.path.exists(root) and download):

            print("Cat dataset is not found, it will be downloaded to datasets/cats_64. (24 MB)")

            cats_64_url = "https://drive.google.com/uc?id=19vLd3nuT3amuW4xlN7kTXoSYqWZlRRqx&export=download"
            urllib.request.urlretrieve(cats_64_url, os.path.join("datasets", "cats.tar.gz"))

            print("Cat dataset is downloaded, now starting extracting the file.")
            cats_tar = tarfile.open(os.path.join("datasets", "cats.tar.gz"))
            cats_tar.extractall("datasets") 
            cats_tar.close()
            print("Extraction is completed.")

        self.files = sorted(glob(root + '/*.png')) + sorted(glob(root + '/*.jpg'))

        self.transforms = transforms

    def __getitem__(self,index):

        return self.transforms(Image.open(self.files[index]))

    def __len__(self):

        return len(self.files)



def get_model(args):  
    """
    Returns the generator and discriminator models for the given model architecture and parameters such as no_BN, all_tanh and spec_norm.

        StandardCNN is the architecture described in the appendices I.1 of the paper, and DCGAN_64 is in appendices I.2.
        
    """
    #  

    if(args.model == "standart_cnn"):

        return (StandardCNN_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh).to(args.device), 
                StandardCNN_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm).to(args.device))

    if(args.model == "dcgan_64"):

        return (DCGAN_64_Generator(no_BN = args.no_BN, all_tanh=args.all_tanh).to(args.device), 
                DCGAN_64_Discriminator(no_BN = args.no_BN, all_tanh=args.all_tanh, spec_norm = args.spec_norm).to(args.device))


def get_loss(loss_type):
    """
    Returns the generator and discriminator losses for the given loss type.

        Relativistic generator losses uses discriminator output for the real samples.

        Relativistic average losses uses the average of discriminator outputs for both real and fake samples.

        Pre-calculated gradient penalty term is added to the discriminator losses using gradient penalty.

    """
     

    if(loss_type == "sgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss(C_real,ones) + loss(C_fake,zeros))

        def gen_loss(C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss(C_fake,ones)

    elif(loss_type == "rsgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_real-C_fake),ones)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_fake-C_real),ones)
        

    elif(loss_type == "rasgan"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 

    elif(loss_type == "lsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss(C_real, zeros) + loss(C_fake, ones))

        def gen_loss(C_fake):
            
            zeros = torch.zeros_like(C_fake)
            return loss(C_fake,zeros)

    elif(loss_type == "ralsgan"):

        loss = nn.MSELoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_real-C_avg_fake), ones) + loss((C_fake-C_avg_real), -ones))

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_fake-C_avg_real), ones) + loss((C_real-C_avg_fake),-ones)) 

    elif(loss_type == "hingegan"):

        def disc_loss(C_real, C_fake):

            ones = torch.ones_like(C_fake)
            return (clamp((ones-C_real), min=0).mean() + clamp((C_fake+ones), min=0).mean())

        def gen_loss(C_fake):
            
            return -C_fake.mean()

    elif(loss_type == "rahingegan"):

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (clamp((ones - C_real + C_avg_fake), min=0).mean() + clamp((ones + C_fake-C_avg_real), min=0).mean())

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            return (clamp((ones - C_fake + C_avg_real), min=0).mean() + clamp((ones + C_real - C_avg_fake), min=0).mean())

    elif(loss_type == "wgan-gp"):

        def disc_loss(C_real, C_fake, grad_pen):
            
            return (-C_real.mean() + C_fake.mean() + grad_pen)

        def gen_loss(C_fake):
            
            return -C_fake.mean()

    elif(loss_type == "rsgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, grad_pen):
            
            ones = torch.ones_like(C_fake)
            return (loss((C_real - C_fake),ones) + grad_pen)

        def gen_loss(C_real, C_fake):
            
            ones = torch.ones_like(C_fake)
            return loss((C_fake-C_real), ones)

    elif(loss_type == "rasgan-gp"):

        loss = nn.BCEWithLogitsLoss()

        def disc_loss(C_real, C_fake, C_avg_real, C_avg_fake, grad_pen):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),ones) + loss((C_fake-C_avg_real),zeros) + grad_pen)

        def gen_loss(C_real, C_fake, C_avg_real, C_avg_fake):
            
            ones = torch.ones_like(C_fake)
            zeros = torch.zeros_like(C_fake)
            return (loss((C_real-C_avg_fake),zeros) + loss((C_fake-C_avg_real),ones)) 


    return gen_loss,disc_loss

def grad_penalty(discriminator, x_hat, Lambda):
    """ 
    Calculates gradient penalty given the interpolated input x_hat.

    Lambda is the gradient penalty coefficient.

    """

    x_hat.requires_grad_(True)
    disc_out = discriminator(x_hat)
    grads = grad(outputs=disc_out, inputs=x_hat,
                 grad_outputs = torch.ones_like(disc_out),
                 create_graph=True)[0].view(x_hat.size(0),-1)

    return Lambda * torch.mean((grads.norm(p=2, dim=1) - 1)**2)



def get_dataset(dataset):
    """
    Returns the Dataset object of the given dataset, "cifar10" or "cat"
    
        For "cifar10", the class torchvision.datasets.CIFAR10 is used. It automatically downloads the dataset if it is not downloaded before.

        For "cat", the images in the folder "./datasets/cat_64" will be used for creating the dataset. If the folder does not exist, it will be automatically downloaded. 

    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if(dataset == "cifar10"):

        return CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)

    if(dataset == "cat"):

        return Dataset(root=os.path.join("datasets", "cats_64"), transforms=transform, download=True)
    

def cycle(iterable, dset, device):
    """
    Restarts the dataloader iteration if it is iterated completely.

    Returns an iterable object of the batches which are sent to the preferred device(cpu or gpu).

        Returns x[0] for "cifar10" dataset since it returns a list of [images, labels] for its batches.

    """
    while True:
        for x in iterable:
            yield (x[0].to(device) if dset=='cifar10' else x.to(device))
            

def is_negative(value):
    """
    Checks if the given value as the argument is negative or not. If it is negative, give an error.

        Used for checking negative iteration frequency arguments.

    """

    if int(value) < 0:
        raise ArgumentTypeError(f"{value} should be non-negative")
    return int(value)


def sample_fid(generator, it, args, batch_size=500):
    """
    Generates samples to be used for calculating FID and saves them as a compressed numpy array.

        The number of samples going to be generated is equal to the number of images in the training set. (args.fid_sample)

    """

    generator.eval()

    with torch.no_grad():

        for i in range(0,args.fid_sample, batch_size):

            if(args.print_iter):
                sys.stdout.write(f"\rsaving {i}/{args.fid_sample}")

            if(args.fid_sample < batch_size+i):
                batch_size = args.fid_sample-i

            generated_samples = (generator(torch.randn(size=(batch_size,128,1,1), device=generator.device))+1)*127.5 

            if(i == 0):
                arr = np.round_(generated_samples.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)
            else:
                arr = np.concatenate((arr, np.round_(generated_samples.cpu().permute(0,2,3,1).numpy()).astype(np.uint8)), axis=0)
            
        np.savez_compressed(f"samples/{args.dataset}_{args.loss_type}_n_d_{args.d_iter}_b_size_{args.batch_size}_lr_{args.lr}_{it+1}", images=arr)

    generator.train()


