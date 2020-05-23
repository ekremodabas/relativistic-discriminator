import ipywidgets as widgets
import numpy as np
from IPython.display import display
from argparse import ArgumentParser
import os
from subprocess import call


download_dict = {
    "cat_lsgan.tar.gz":   "https://drive.google.com/uc?id=1oTzTFQvcf1vXjCkh8HGK0lGbaNkS7-vo&export=download",
    "cat_rasgan-gp.tar.gz":   "https://drive.google.com/uc?id=1Tfu6l7iGQvU8snrua84e-doULWMw7SNa&export=download",
    "cat_rsgan-gp.tar.gz":   "https://drive.google.com/uc?id=1zjry3GXA6H4efdy3tZlU6-IORXAouGNv&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1xHb7kz1fmHwk0gXsArtcbNa4oLTLz1sN&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1xfKBB_wz94am9fDWrkhYcTsc8bR3Tz3U&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1y8V8SFoeRErI_oDAaxK4sqROklTPO1B8&export=download",
    "cat_ralsgan.tar.gz":   "https://drive.google.com/uc?id=1zPDVIv1txV_WFSMo1OWT0-o0wYhUxaPb&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1za2MEwbN4GsxjLTV0sHf0EfNxRRB8gs-&export=download",
    "gen_cifar10_hingegan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1om3KtU1Q-5k7SXq-jKYgC9z9K1RSPyz4&export=download",
    "gen_cifar10_rsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1pd8SwAXU7vGWiG_Z3gmN3c-O-krSL8E-&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1qMsPo-LSrH7ta2wnjo6itCpGQqlTtl_M&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1szHr_vaLGK3_pr5YOfKNsCxyx40wzSaW&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1tffHN2nL-PXXpSs6SYAL1LxrrFKi9QnE&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1uq8rhKYpv08gwm8iJSYeV34WL_Fo3irD&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1kQfcD3F5L1RUftCfGQNy6SOFdwMmiCK5&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1mTwPFOp9bSxmzNA4XfVYqd7boVblVQvL&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1nDX-k5AUUdfSEEniifrlwWUQqm8hQcw3&export=download",
    "gen_cifar10_rsgan-gp_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1oRHzNh3rOCyEExpEwViI_P8rn_a1xmL4&export=download",
    "gen_cifar10_rasgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1oYj0vTyBA96BxI7A7sqH8AnVrYo5Fbu7&export=download",
    "gen_cifar10_rahingegan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1fuKaVvLiGiAWpnGNVOYf9dbgPFLXkFOe&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1g-nZs7fMGkfz5nl7Czn0Aph7Od1W0wzR&export=download",
    "cat_rahingegan.tar.gz":   "https://drive.google.com/uc?id=1goc6gcpghzZB9jf7XbswUNbsTYtoam1i&export=download",
    "gen_cifar10_sgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1j696N2dZ4YX8xqlhAnoqeN238fs3EMrb&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1jae9XPDL9jWYji73IVfjbd-ip4CoMJc9&export=download",
    "cat_sgan.tar.gz":   "https://drive.google.com/uc?id=1jm9plvOIbHaw8auWlmQFtfoKZuSU6nz8&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1aGf31oW76pWXOgGDspjjoYToFT0QC1MH&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1ax_iOQj1RVddTPg1_DboAQturRxhvZ1W&export=download",
    "gen_cifar10_wgan-gp_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1c0KnrK1KXa0qOlPls0eT-7f27NLoHzzg&export=download",
    "gen_cifar10_lsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1dJsKOwqVU00CdYvgBpq7CLo2aXahQvLp&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1dm_V6T-B7XLPCHgmEOpehi_ZzcVT0E9i&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1XtJ_mayaOPHHe3KCuMe-Zaa-M_iJjQ9K&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1ZMPOFuaAXUib-aePVpCYfPvhkKkv_onJ&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1Zqtoz0TRyjpxi5POgVG6TXvH1gnFKk4o&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1_DzU04_k9FsilMkTTXauw949z4Igz2M6&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1_fXTWBGpDKL-undp3kmQSWzdrJk9f_0m&export=download",
    "cat_hingegan.tar.gz":   "https://drive.google.com/uc?id=1N9_FvAFpvx2RcV-CIVIPldF5Wsr4zR16&export=download",
    "cat_rasgan.tar.gz":   "https://drive.google.com/uc?id=1PkE_FK3u5SLUIWqyJdFxlDmbeGVNgNjb&export=download",
    "gen_cifar10_ralsgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1S3_Katg1Gr60wPWiPb2adPsUp8vquDve&export=download",
    "gen_cifar10_rasgan_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.pth":   "https://drive.google.com/uc?id=1SnSBVb4Lwd_4p7F9ulHAyVAxwmet6FwV&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1XBsckZfpLquu-CCexvPPmQReGwvNYNlA&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1KTdmBlfIn3KUTHqCPkx95c5cqrIq6E3P&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1KqzU8Ooblzu8FzycDnemOrjL-6sgu_qt&export=download",
    "gen_cifar10_hingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1L5iuO4EM7nGiZuDFFKcfX40Z0Y5d0zSy&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1LY7G_Puv7EvPh66EOudjrVLSBokC1n8n&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1MO0Yjkr_56NPBWpIgfd7LO2F-SSJ1nJL&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.pth":   "https://drive.google.com/uc?id=1FaXYHfePO-1kw2hwhAjeqKhT194VIAP7&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1H2_Eb6PwfZZ4Hb7UQ-J2dGEBN2AkUmpC&export=download",
    "gen_cifar10_rahingegan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1HFGJxWZo_myyRnxtV6jfQ295O1GxC-Xh&export=download",
    "gen_cifar10_rsgan-gp_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1JGpjujPMH0Ip-f604dJdcZqhGtduYel5&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1Jn9ScH1DHWH-RrpJ1j98QRVmYeAVbIgQ&export=download",
    "cat_rsgan.tar.gz":   "https://drive.google.com/uc?id=17o2lEbGg1hv0xsikHPlqkKn6Y2x3uRHW&export=download",
    "gen_cifar10_rasgan_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=18Z808e5CqZq2jXNMQutKvSVpwUWpX2t8&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1AQA1lMsAeX_0hre5GnzMHZL6-Ju3I5yW&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=1BM5NulMk1NTsKiSBwJkyZ1OJHXt1gCln&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=1E7qeqVMADcITOU_BW0EDE_26idkxDs8_&export=download",
    "gen_cifar10_sgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=1EQi5kqPW23frH1uIoBLEZ8SR4woo5_GY&export=download",
    "gen_cifar10_ralsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=12RF5y9LYMA7kPXwGl1GZy_gDjbdxzWDJ&export=download",
    "gen_cifar10_lsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.pth":   "https://drive.google.com/uc?id=14nITdHjaalbmiHQIRF4_kDp6xtoJyfLm&export=download",
    "gen_cifar10_rsgan_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.pth":   "https://drive.google.com/uc?id=15-ymRj0Y8xf8D2FNDr8VN56SSD1NfhXm&export=download",
    "gen_cifar10_wgan-gp_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.pth":   "https://drive.google.com/uc?id=15D1IwouiORzsXSl7xeF1w4q5p0doGk7Z&export=download"

}

def def_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--d_iter', type=int, default=1, help='the number of iterations to train the discriminator before training the generator')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type, "cifar10" or "cat"')
    parser.add_argument('--model', type=str, default='standart_cnn', help='model architecture, "standard_cnn" or "dcgan_64"')
    parser.add_argument('--loss_type', type=str, default='sgan', help='loss type, "sgan", "rsgan", "rasgan", "lsgan", "ralsgan", "hingegan", "rahingegan", "wgan-gp", "rsgan-gp" or "rasgan-gp"')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for the discriminator and the generator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 value of adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 value of adam')
    parser.add_argument('--spec_norm', type=bool, default=False, help = 'spectral normalization for the discriminator')
    parser.add_argument('--no_BN', type=bool, default=False, help = 'do not use batchnorm for any of the models')
    parser.add_argument('--all_tanh', type=bool, default=False, help = 'use tanh for all activations of the models')
    parser.add_argument('--lambd', type=int, default=10, help='coefficient for gradient penalty')
    return parser.parse_args([])
   
def reproduce_results(experiment, part, use_pretrained):

    cat_ids = np.array(["All", "SGAN Experiment", "RSGAN Experiment", "RaSGAN Experiment", "LSGAN Experiment", "RaLSGAN Experiment", "HingeGAN Experiment", "RaHingeGAN Experiment", "RSGAN-GP Experiment", "RaSGAN-GP Experiment"])

    cifar10_ids = np.array(["All", "SGAN Experiment 1", "SGAN Experiment 2", "RSGAN Experiment 1", "RSGAN Experiment 2", "RaSGAN Experiment 1", "RaSGAN Experiment 2", "LSGAN Experiment 1", "LSGAN Experiment 2", "RaLSGAN Experiment 1", "RaLSGAN Experiment 2", "HingeGAN Experiment 1", "HingeGAN Experiment 2", "RaHingeGAN Experiment 1", "RaHingeGAN Experiment 2", "WGAN-GP Experiment 1", "WGAN-GP Experiment 2", "RSGAN-GP Experiment 1", "RSGAN-GP Experiment 2", "RaSGAN-GP Experiment 1", "RaSGAN-GP Experiment 2"])

    unstable_ids = np.array(["All", "SGAN lr = 0.001","SGAN Beta = (0.9, 0.9)","SGAN Remove BatchNorms","SGAN All Activations Tanh","RSGAN lr = 0.001","RSGAN Beta = (0.9, 0.9)","RSGAN Remove BatchNorms","RSGAN All Activations Tanh","RaSGAN lr = 0.001","RaSGAN Beta = (0.9, 0.9)","RaSGAN Remove BatchNorms","RaSGAN All Activations Tanh","LSGAN lr = 0.001","LSGAN Beta = (0.9, 0.9)","LSGAN Remove BatchNorms","LSGAN All Activations Tanh","RaLSGAN lr = 0.001","RaLSGAN Beta = (0.9, 0.9)","RaLSGAN Remove BatchNorms","RaLSGAN All Activations Tanh","HingeGAN lr = 0.001","HingeGAN Beta = (0.9, 0.9)","HingeGAN Remove BatchNorms","HingeGAN All Activations Tanh","RaHingeGAN lr = 0.001","RaHingeGAN Beta = (0.9, 0.9)","RaHingeGAN Remove BatchNorms","RaHingeGAN All Activations Tanh","WGAN-GP lr = 0.001","WGAN-GP Beta = (0.9, 0.9)","WGAN-GP Remove BatchNorms","WGAN-GP All Activations Tanh"])

    part = np.array(part)

    args = ArgumentParser().parse_args([])

    if(experiment == 0): # cifar

        if(part[0] == 0):

            print("Going to reproduce every CIFAR10 experiment.")

            num_exp = 19

            part = np.arange(1,20)

        else:

            print("Going to reproduce ", end = '')

            reproduced = cifar10_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for CIFAR10.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.spec_norm = True
            args.model = "standard_cnn"
            args.batch_size = 64
            args.fid_sample = 50000
            # args.device

            os.makedirs("samples", exist_ok = True)

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in cifar10_ids[part]:

                print(f"Downloading model for {exp}")

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = int(exp_split[2])

                the_key = ""

                if(exp_type == 1):

                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_64"

                else:
                    args.d_iter = 5
                    args.beta2 = 0.9
                    args.lr = 0.0001
                    model_text = "_n_d_5_b1_0.5_b2_0.9_b_size_64"

                for key in download_dict.keys():

                    if(key.startswith(f"gen_cifar10_{loss.lower()}{model_text}")):

                        url = download_dict[key]
                        the_key = key
                        break

                urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                args.loss_type = loss.lower()

                Generator, _ = get_model(args)

                Generator.load_state_dict(torch.load(os.path.join("models", f"{the_key}"), map_location=args.device))

                print(f"Creating samples for {exp}")

                sample_name = the_key[:-3]+"npz"

                sample_fid(Generator, 99999, args)

                print(f"Calculating the FID between the generated samples and the real samples...")

                # run fid calculation and write it to the file

                print(f"Calculated the FID, re-run the read_calculations to see the result.")


        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_cifar10.sh", "w+")

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in cifar10_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = int(exp_split[2])

                if(exp_type == 1):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_100000.npz")
                    training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --spec_norm True\n")

                else:

                    samples.append(f"cifar10_{loss.lower()}_n_d_5_b1_0.5_b2_0.9_b_size_64_lr_0.0001_100000.npz")
                    training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --spec_norm True --lr 0.0001 --beta2 0.9 --d_iter 5\n")

                
            print(f"Starting training the models.")

            run("./reproduce_cifar10.sh")

            # run the fid calculator | samples

            print(f"Calculated the FID, re-run the read_calculations to see the result.")




    elif(experiment== 1): # cat

        if(part[0] == 0):

            print("Going to reproduce every CAT 64x64 experiment.")

            num_exp = 9

            part = np.arange(1,10)

        else:

            print("Going to reproduce ", end = '')

            reproduced = cat_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for CAT 64x64.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.model = "dcgan_64"
            args.batch_size = 64
            args.fid_sample = 9303
            # args.device

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in cat_ids[part]:

                print(f"Downloading model for {exp}")

                exp_split = exp.split(" ")

                loss = exp_split[0]

                the_key = ""

                for key in download_dict.keys():

                    if(key.startswith(f"cat_{loss.lower()}")):

                        url = download_dict[key]
                        the_key = key
                        break

                urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                print("Extracting the models...")
                cats_tar = tarfile.open(os.path.join("models", f"{the_key}"))
                cats_tar.extractall("models") 
                cats_tar.close()
                print("Extraction is completed.")

                Generator, _ = get_model(args)

                for it in range(20000,100001,10000):

                    Generator.load_state_dict(torch.load(os.path.join("models", f"cat_{loss.lower()}", f"gen_cat_{loss.lower()}_n_d_1_b_size_64_lr_0.0002_{it}.pth"), map_location=args.device))

                    print(f"Creating samples for {exp} {it}/100000")

                    sample = f"cat_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_{it}.npz"

                    sample_fid(Generator, it-1, args)

                    print(f"Calculating the FID between the generated samples and the real samples...")

                    # run fid calculation and write it to the file

                    print(f"Calculated the FID, re-run the read_calculations to see the result.")

        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_cat.sh", "w+")

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in cat_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                for it in range(20000,100001,10000):

                    samples.append(f"cat_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_64_lr_0.0002_{it}.npz")

                training_script.write(f"python main.py --loss_type {loss.lower()} --batch_size 64 --dataset cat --model dcgan_64 --fid_iter 10000 --save_model 10000")
            
            print(f"Starting training the models.")

            run("./reproduce_cat.sh")

            # run the fid calculator

            print(f"Calculated the FID, re-run the read_calculations to see the result.")


    else: # unstable

        if(part[0] == 0):

            print("Going to reproduce every unstable experiment.")

            num_exp = 32

            part = np.arange(1,33)

        else:

            print("Going to reproduce ", end = '')

            reproduced = unstable_ids[part]

            for i in range(len(reproduced)):
                print( (f"{reproduced[i]}, " if i+1 != len(reproduced) else f"{reproduced[i]} "), end='')

            print("for unstable experiments.")

            num_exp = len(reproduced)


        if(use_pretrained):

            args = def_args()
            args.fid_sample = 50000

            print(f"Downloading {num_exp}" + (" models "  if num_exp!=1 else " model ") + "to models folder...")

            for exp in unstable_ids[part]:

                print(f"Downloading model for {exp}")

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = exp_split[1]

                the_key = ""

                if(exp_type == 'lr'):

                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001"
                    args.lr = 0.001

                elif(exp_type == 'Beta'):

                    model_text = "_n_d_1_b1_0.9_b2_0.9_b_size_32_"
                    args.beta1=0.9
                    args.beta2=0.9
                    
                elif(exp_type == 'Remove'):

                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN"
                    args.no_BN = True

                elif(exp_type == 'All'):

                    model_text = "_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh"
                    args.all_tanh = True

                for key in download_dict.keys():                    

                    if(key.startswith(f"gen_cifar10_{loss.lower()}{model_text}")):

                        url = download_dict[key]
                        the_key = key
                        break

                urllib.request.urlretrieve(url, os.path.join("models", f"{the_key}"))

                args.loss_type = loss.lower()

                Generator, _ = get_model(args)

                Generator.load_state_dict(torch.load(os.path.join("models", f"{the_key}"), map_location=args.device))

                print(f"Creating samples for {exp}")

                sample = the_key[:-3]+"npz"

                sample_fid(Generator, 100000, args)

                print(f"Calculating the FID between the generated samples and the real samples...")

                # run fid calculation and write it to the file

                print(f"Calculated the FID, re-run the read_calculations to see the result.")

        else:

            print(f"Creating training script for {num_exp}" + (" models"  if num_exp!=1 else " model") + ".")

            training_script = open("reproduce_unstable.sh", "w+")

            training_script.write("#!/usr/bin/env bash\n")

            samples = []

            for exp in unstable_ids[part]:

                exp_split = exp.split(" ")

                loss = exp_split[0]

                exp_type = exp_split[1]

                training_script.write(f"python main.py --loss_type {loss.lower()} ")

                if(exp_type == 'lr'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.001_100000.npz")

                    training_script.write("--lr 0.001\n")

                elif(exp_type == 'Beta'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.9_b2_0.9_b_size_32_lr_0.0002_100000.npz")

                    training_script.write("--beta1 0.9 --beta2 0.9\n")
                    
                elif(exp_type == 'Remove'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_noBN.npz")

                    training_script.write("--no_BN True\n")

                elif(exp_type == 'All'):

                    samples.append(f"cifar10_{loss.lower()}_n_d_1_b1_0.5_b2_0.999_b_size_32_lr_0.0002_100000_alltanh.npz")

                    training_script.write("--all_tanh True\n")

            print(f"Starting training the models.")

            run("./reproduce_unstable.sh")

            # run the fid calculator | samples

            print(f"Calculated the FID, re-run the read_calculations to see the result.")

            
    print(experiment, part, use_pretrained)



def create_markdown_cifar(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|$lr$<span style="font-weight:normal">= .0002</span><br>$\beta$ <span style="font-weight:normal">= (0.5,0.999)</span><br>$n_D$ <span style="font-weight:normal">= 1</span>|$lr$<span style="font-weight:normal">= .0001</span><br>$\beta$ <span style="font-weight:normal">= (0.5,0.9)</span><br>$n_D$ <span style="font-weight:normal">= 5</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|" + "\n"
    mins_0, mins_1 = [], []
    for key in fid_dict.keys():
        mins_0.append(fid_dict[key][0])
        mins_1.append(fid_dict[key][1])
    mins_0, mins_1 = min(mins_0), min(mins_1)
    for key in fid_dict.keys():
        markdown += f"|{key}" + \
                    (f"|{fid_dict[key][0]:.{precision}f}|" if(fid_dict[key][0]!=mins_0) else f"|**{fid_dict[key][0]:.{precision}f}**|") + \
                    ((f"{fid_dict[key][1]:.{precision}f}" if(fid_dict[key][1]!=mins_1) else f"**{fid_dict[key][1]:.{precision}f}**") if key!="RaSGAN-GP" else "") + "|\n"
    return markdown

def create_markdown_cat(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|<span style="font-weight:normal">Minimum FID</span>|<span style="font-weight:normal">Maximum FID</span>|<span style="font-weight:normal">Mean of FIDs</span>|<span style="font-weight:normal">SD of FIDs</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|:-:|:-:|" + "\n"
    mins, maxs, devs, means = [], [], [], []
    for key in fid_dict.keys():
        mins.append(min(fid_dict[key]))
        maxs.append(max(fid_dict[key]))
        devs.append(np.std(fid_dict[key]))
        means.append(fid_dict[key].mean())
    mins, maxs, devs, means = min(mins), min(maxs), min(devs), min(means)
    for key in fid_dict.keys():
        markdown += f"|{key}" +  \
                    (f"|{min(fid_dict[key]):.{precision}f}" if(min(fid_dict[key])!=mins) else f"|**{min(fid_dict[key]):.{precision}f}**") + \
                    (f"|{max(fid_dict[key]):.{precision}f}" if(max(fid_dict[key])!=maxs) else f"|**{max(fid_dict[key]):.{precision}f}**") + \
                    (f"|{fid_dict[key].mean():.{precision}f}"  if(fid_dict[key].mean()!=means) else f"|**{fid_dict[key].mean():.{precision}f}**") + \
                    (f"|{np.std(fid_dict[key]):.{precision}f}|\n" if(np.std(fid_dict[key])!=devs) else f"|**{np.std(fid_dict[key]):.{precision}f}**|\n")
    return markdown

def create_markdown_unstable(fid_dict, precision=3):
    markdown = r'|<span style="font-weight:normal">Loss Type</span>|$lr$<span style="font-weight:normal">= .001</span>|$\beta$<span style="font-weight:normal">=(0.9,0.9)</span>|<span style="font-weight:normal">No BN</span>|<span style="font-weight:normal">Tanh Activations</span>|' + "\n"
    markdown += r"|:-:|:-:|:-:|:-:|:-:|" + "\n"
    mins_0, mins_1, mins_2, mins_3 = [], [], [], []
    for key in fid_dict.keys():
        mins_0.append(fid_dict[key]['lr'])
        mins_1.append(fid_dict[key]['beta'])
        mins_2.append(fid_dict[key]['bn'])
        mins_3.append(fid_dict[key]['tanh'])
    mins_0, mins_1, mins_2, mins_3 = min(mins_0), min(mins_1), min(mins_2), min(mins_3)
    for key in fid_dict.keys():
        markdown += f"|{key}" +  \
                    (f"|{fid_dict[key]['lr']:.{precision}f}" if(fid_dict[key]['lr']!=mins_0) else f"|**{fid_dict[key]['lr']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['beta']:.{precision}f}" if(fid_dict[key]['beta']!=mins_1) else f"|**{fid_dict[key]['beta']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['bn']:.{precision}f}"  if(fid_dict[key]['bn']!=mins_2) else f"|**{fid_dict[key]['bn']:.{precision}f}**") + \
                    (f"|{fid_dict[key]['tanh']:.{precision}f}|\n" if(fid_dict[key]['tanh']!=mins_3) else f"|**{fid_dict[key]['tanh']:.{precision}f}**|\n")
    return markdown


def interaction():
    layout = widgets.Layout(width='auto')
    style = {'description_width': 'initial'}
    use_pretrained = widgets.Checkbox(value=True,indent = False, style=style, layout=layout, description='Use pre-trained models to calculate FIDs (Will re-train if not checked)')

    experiment = widgets.ToggleButtons(options=[('Cifar 10 Experiments', 0), ('Cat 64x64 Experiments', 1), ('Unstable Experiments', 2)], style=style, layout=layout,value=0,description='Experiment:')

    part =  widgets.SelectMultiple(options=[("All", 0),("SGAN Experiment 1",1),("SGAN Experiment 2",2),("RSGAN Experiment 1",3),("RSGAN Experiment 2",4),("RaSGAN Experiment 1",5),("RaSGAN Experiment 2",6),("LSGAN Experiment 1",7),("LSGAN Experiment 2",8),("RaLSGAN Experiment 1",9),("RaLSGAN Experiment 2",10),("HingeGAN Experiment 1",11),("HingeGAN Experiment 2",12),("RaHingeGAN Experiment 1",13),("RaHingeGAN Experiment 2",14),("WGAN-GP Experiment 1",15),("WGAN-GP Experiment 2",16),("RSGAN-GP Experiment 1",17),("RSGAN-GP Experiment 2",18),("RaSGAN-GP Experiment 1",19)],
                            value=[0],rows=8,disabled=False, style = {'description_width': 'initial'},layout = layout,description='Reproduce Selected Parts (Ctrl+Click for Multiple Selection):')

    def update_parts(*args):
        parts = [[("All", 0),("SGAN Experiment 1",1),("SGAN Experiment 2",2),("RSGAN Experiment 1",3),("RSGAN Experiment 2",4),("RaSGAN Experiment 1",5),("RaSGAN Experiment 2",6),("LSGAN Experiment 1",7),("LSGAN Experiment 2",8),("RaLSGAN Experiment 1",9),("RaLSGAN Experiment 2",10),("HingeGAN Experiment 1",11),("HingeGAN Experiment 2",12),("RaHingeGAN Experiment 1",13),("RaHingeGAN Experiment 2",14),("WGAN-GP Experiment 1",15),("WGAN-GP Experiment 2",16),("RSGAN-GP Experiment 1",17),("RSGAN-GP Experiment 2",18),("RaSGAN-GP Experiment 1",19)], [("All", 0),("SGAN Experiment"        ,1),("RSGAN Experiment"        ,2),("RaSGAN Experiment"        ,3),("LSGAN Experiment"        ,4),("RaLSGAN Experiment"      ,5),("HingeGAN Experiment"      ,6),("RaHingeGAN Experiment"   ,7),("RSGAN-GP Experiment"      ,8),("RaSGAN-GP Experiment"     ,9)], [("All", 0),("SGAN lr = 0.001",1),("SGAN Beta = (0.9, 0.9)",2),("SGAN Remove BatchNorms",3),("SGAN All Activations Tanh",4),("RSGAN lr = 0.001",5),("RSGAN Beta = (0.9, 0.9)",6),("RSGAN Remove BatchNorms",7),("RSGAN All Activations Tanh",8),("RaSGAN lr = 0.001",9),("RaSGAN Beta = (0.9, 0.9)",10),("RaSGAN Remove BatchNorms",11),("RaSGAN All Activations Tanh",12),("LSGAN lr = 0.001",13),("LSGAN Beta = (0.9, 0.9)",14),("LSGAN Remove BatchNorms",15),("LSGAN All Activations Tanh",16),("RaLSGAN lr = 0.001",17),("RaLSGAN Beta = (0.9, 0.9)",18),("RaLSGAN Remove BatchNorms",19),("RaLSGAN All Activations Tanh",20),("HingeGAN lr = 0.001",21),("HingeGAN Beta = (0.9, 0.9)",22),("HingeGAN Remove BatchNorms",23),("HingeGAN All Activations Tanh",24),("RaHingeGAN lr = 0.001",25),("RaHingeGAN Beta = (0.9, 0.9)",26),("RaHingeGAN Remove BatchNorms",27),("RaHingeGAN All Activations Tanh",28),("WGAN-GP lr = 0.001",29),("WGAN-GP Beta = (0.9, 0.9)",30),("WGAN-GP Remove BatchNorms",31),("WGAN-GP All Activations Tanh",32)]]
        part.options = parts[experiment.value]
    part.observe(update_parts, 'value')

    info = widgets.Text(value='Select experiments to reproduce and click "Run Interact" button (Parts will be updated when clicked on after changing the experiment)',placeholder='',description='',disabled=True, layout=layout)

    
    selecter = widgets.interactive(reproduce_results, {'manual': True}, experiment=experiment, part = part, use_pretrained = use_pretrained)

    return info, selecter



