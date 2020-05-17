#!/usr/bin/python3
import argparse
import itertools
from torch.utils.data import DataLoader
import torch
# from torchvision.utils import save_image
# from utils import LambdaLR, weights_init_normal
# from utils import ImageDataset
from utils import get_model, get_loss, get_dataset, cycle, grad_penalty
import os
from matplotlib.pyplot import imshow, show
import numpy as np
import sys
import time
import random

from utils import sampl, sampl_fid

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', type=int, default=100000, help='total number of iterations')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--seed', type=int, default=1, help='default seed for torch, use -1 to train with random seed')
parser.add_argument('--d_iter', type=int, default=1, help='the number of discriminator updates before updating the generator')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type, "cifar10" or "cat"')
parser.add_argument('--model', type=str, default='standart_cnn', help='model, "standart_cnn" or "dcgan_64"')
parser.add_argument('--model_type', type=str, default='sgan', help='model type, "sgan", "rsgan", "rasgan", "lsgan", "ralsgan", "hingegan", "rahingegan", "wgan-gp", "rsgan-gp" or "rasgan-gp"')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
parser.add_argument('--lambd', type=int, default=10, help='gradient penalty lambda')
parser.add_argument('--n_workers', type=int, default=4, help='number of cpu threads for data loader')
parser.add_argument('--info', type=str, default = '', help = 'information about training')
parser.add_argument('--spec_norm', type=bool, default=False, help = 'spectral normalization for discriminator')
parser.add_argument('--no_BN', type=bool, default=False, help = 'no batchnorm')
parser.add_argument('--all_tanh', type=bool, default=False, help = 'tanh for all activations')
parser.add_argument('--fid_iter', type=int, default=100000, help='number of cpu threads for data loader')


args = parser.parse_args()


# os.makedirs(os.path.join("models", f"{info}"), exist_ok=True)
# os.makedirs(os.path.join("outputs", "training", f"{info}"), exist_ok=True)
# os.makedirs(os.path.join("losses","training", f"{info}"), exist_ok=True)

if(args.seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# no non cuda version

Generator, Discriminator = get_model(args)

Generator.cuda()
Discriminator.cuda()

optimizer_G = torch.optim.Adam(params=Generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(params=Discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

#model type, "sgan", "rsgan", "rasgan", "lsgan", "ralsgan", "hingegan", "rahingegan", 
#"wgan-gp", "rsgan-gp" or "rasgan-gp"')

gen_loss, disc_loss = get_loss(args.model_type)

gradient_pen = args.model_type in ["wgan-gp", "rsgan-gp", "rasgan-gp"]
relativistic = args.model_type in ["rsgan", "rasgan", "ralsgan", "rahingegan", "rsgan-gp","rasgan-gp"]
average = args.model_type in ["rasgan", "ralsgan", "rahingegan", "rasgan-gp"]


dataset = get_dataset(args.dataset)

losses = open(f"losses/{args.dataset}_{args.model_type}_n_d_{args.d_iter}_b_size_{args.batch_size}_lr_{args.lr}.txt", "a+")

# is shuffle false?
dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size ,shuffle=True,  num_workers=args.n_workers)

# loader_iter = iter(dataloader)
loader_iter = iter(cycle(dataloader, args.dataset))



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# sample_noise = torch.randn(10,128).cuda()
st_time = time.time()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


for i in range(0,args.n_iter):#, args.batch_size):

    for t in range(args.d_iter):

        real = next(loader_iter).cuda()

        noise = torch.randn(real.size(0),128,1,1).cuda()

        optimizer_D.zero_grad()

        fake = Generator(noise).detach()
        
        loss_args_D = [Discriminator(real), Discriminator(fake)]
        
#         print(loss_args_D[0].size())
#         print("a")
        
        if(average):

            loss_args_D += [loss_args_D[0].mean(), loss_args_D[1].mean()]

        if(gradient_pen):

            random_num = torch.rand(real.size(0),1,1,1).cuda()

            x_hat = random_num * real + (1-random_num) * fake

            loss_args_D += [grad_penalty(Discriminator, x_hat, args.lambd)]

        loss_D = disc_loss( *loss_args_D )

        loss_D.backward()

        optimizer_D.step()

    real = next(loader_iter).cuda()

    noise = torch.randn(real.size(0), 128,1,1).cuda()

    optimizer_G.zero_grad()

    fake = Generator(noise)
    
    loss_args_G = [Discriminator(fake)]

    if(relativistic):

        loss_args_G = [Discriminator(real)] + loss_args_G

    if(average):
                                                                # is it detached?
        loss_args_G += [Discriminator(real).mean(), Discriminator(fake).mean()]

    loss_G = gen_loss( *loss_args_G )

    loss_G.backward()

    optimizer_G.step()

    
#     if(i % 4096 == 0):
#         sampl(sample_noise, Generator, i)
    
    
    if(i%50 == 0):
        losses.write(f"{i}/{args.n_iter} loss_D {loss_D:.4f} loss_G {loss_G:.4f}\n")
    if(i%1000 == 0 ):
        print(f"time {time.time()-st_time} s {i}")
        st_time= time.time()

    if((i+1) % args.fid_iter == 0):
        
        s_time = time.time()        
        sampl_fid(Generator, i,args)
        print(f"sampling took {time.time()-s_time} s {i}")
        
losses.close()
        


