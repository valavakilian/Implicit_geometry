
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
import pickle

import os
import pickle
import shutil

from generate_cifar import IMBALANCECIFAR10
from generate_mnist import IMBALANCEMNIST

import argparse

from models import *

from criterion import *

from loggers import *

from utils import *


parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_dir', type=str, default='../data', help='path to dataset directory')
parser.add_argument('--save_dir', type=str, default='../saved_logs', help='path to experiment directory')
parser.add_argument('--img_dim', type=int, default=32)
parser.add_argument("--model", type=str, default='ResNet18', choices=['ResNet18', 'MLP'])
parser.add_argument("--dataset", type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'FMNIST'])
parser.add_argument("--K", type=int, default=10)
parser.add_argument('--R', type=int, default=10, help='Imbalance ratio')
parser.add_argument('--rho', type=float, default=0.5, help='Step imbalance cutoff')
parser.add_argument("--n_maj", type=int, default=5000)
parser.add_argument("--augmentation", action='store_true')
parser.add_argument("--versions", type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=350, help='number of training epochs')
parser.add_argument("--gpu", action='store_true')
parser.add_argument('--loss_type', default='CDT', type=str, choices=['wCE', 'CDT', 'LDT'], help='Imbalance loss type')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--lr_decay_epochs', type=float, nargs='+', default = [116, 232], help='learning rate decay epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gamma_values', type=float, nargs='+', default = np.linspace(-1.5, 1.5, num = 13).tolist(), help='List of gamma values')
parser.add_argument("--args_rand", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


log_epoch_list = [1,   3,   5,   7,   9,
                    11,  20,  30,  40,  60,
                    80, 101, 120, 140, 160,
                    180, 201, 220, 235, 245, 250, 260,
                    275, 280, 290, 299, 305, 310, 315, 
                    320, 325, 330, 335, 340, 345, 349, 350]

def main():

    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

    print(vars(args))
    classes = [c for c in range(0, args.K)]
    maj_classes = [c for c in range(0, int(args.K * args.rho))]
    min_classes = [c for c in range(0, args.K) if c not in maj_classes]
    delta_list = [args.R if c in maj_classes else 1 for c in range(0, args.K)]
    n_c_train = [args.n_maj if c in maj_classes else int(args.n_maj // args.R) for c in range(0, args.K)]


    augmentation = "Augment" if args.augmentation else "noAugment"
    general_save_dir = args.save_dir + '/R_' + str(args.R) + "/" + '_'.join([args.model, args.dataset, augmentation]) + "/" + str(args.loss_type) + "/" 
    if not os.path.exists(general_save_dir):
        os.makedirs(general_save_dir, exist_ok=True)

    # ------- Imbalanced dataset --------------------------------------------------------------------------------------------------
    if args.dataset == 'CIFAR10':
        input_ch        = 3
        im_size = 32
        padded_im_size = 32

        test_transforms_list = [ transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        train_transforms_list = test_transforms_list.copy()
        if args.augmentation:
            train_transforms_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
            train_transforms_list.insert(1, transforms.RandomCrop(im_size, padding=4))
            train_transforms_list.insert(2, transforms.RandomHorizontalFlip())
        
        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(test_transforms_list)

        train_dataset = IMBALANCECIFAR10(args.data_dir, imb_type="step",
                                        rand_number=args.args_rand, train=True, download=True,
                                        transform=transform_train, n_c_train_target=n_c_train, classes=classes)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)

    elif args.dataset == 'MNIST':
        input_ch        = 1
        im_size = 28
        padded_im_size = 32

        test_transforms_list = [ transforms.ToTensor(),
                                 transforms.Normalize(0.1307, 0.3081)]
        train_transforms_list = test_transforms_list.copy()
        if args.augmentation:
            train_transforms_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
            train_transforms_list.insert(1, transforms.RandomCrop(im_size, padding=4))
            train_transforms_list.insert(2, transforms.RandomHorizontalFlip())

        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(test_transforms_list)

        train_dataset = IMBALANCEMNIST(args.data_dir, imb_type="step",
                                      rand_number=args.args_rand, train=True, download=True,
                                      transform=transform_train, n_c_train_target=n_c_train,
                                      classes=classes)
        train_dataset.data = torch.tensor(train_dataset.data)

        val_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform_test)
    
    elif args.dataset == 'FMNIST':
        input_ch        = 1
        im_size = 28
        padded_im_size = 32

        test_transforms_list = [ transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))]
        train_transforms_list = test_transforms_list.copy()
        if args.augmentation:
            train_transforms_list.insert(0, transforms.Pad((padded_im_size - im_size)//2))
            train_transforms_list.insert(1, transforms.RandomCrop(im_size, padding=4))
            train_transforms_list.insert(2, transforms.RandomHorizontalFlip())
        

        transform_train = transforms.Compose(train_transforms_list)
        transform_test = transforms.Compose(test_transforms_list)

        train_dataset = IMBALANCEMNIST(args.data_dir, imb_type="step",
                                      rand_number=args.args_rand, train=True, download=True,
                                      transform=transform_train, n_c_train_target=n_c_train,
                                      classes=classes)
        train_dataset.data = torch.tensor(train_dataset.data)

        val_dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform_test)
    

    train_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True, sampler=None)

    analysis_loader = torch.utils.data.DataLoader( train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader( val_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)

    # ------- Main Training -------------------------------------------------------------------------------------------------------
    for version in range(0, args.versions):
        for gamma in args.gamma_values:
            print("-" * 30)
            print("Performing experiment for " + str('_'.join(['gamma', str(gamma), "ver", str(version)])))

            exp_save_path = general_save_dir + '_'.join(['gamma', str(gamma), "ver", str(version)]) + "/"
            exp_complete_flag = exp_save_path + "exp_complete.txt"

            if not os.path.exists(exp_save_path):
                os.makedirs(exp_save_path, exist_ok=True)
            if not os.path.exists(exp_complete_flag):
                shutil.rmtree(exp_save_path)
                os.makedirs(exp_save_path, exist_ok=True)
            else:
                print("Skipping this experiments since flag is set. Please remove flag to rerun this experiment.")
                continue

            # ------- Model -------------------------------------------------------------------------------------------------------
            # Hook for this experiment's features
            class features:
                pass
            def hook(self, input, output):
                features.value = input[0].clone()

            if args.model == "ResNet18":
                model = ResNet18(args.K, input_ch)
                classifier = model.core_model.fc
                classifier.register_forward_hook(hook)
                model = model.to(device)
            if args.model == "MLP":
                model = MLP(hidden = 2048, depth = 6, fc_bias = False)
                classifier = model.fc
                classifier.register_forward_hook(hook)
                model = model.to(device)
            
            # ------- Loss ---------------------------------------------------------------------------------------------------------
            if args.loss_type == 'wCE':
                wCE_weight = [1 if c in maj_classes else args.R**gamma for c in range(0, args.K)]
                wCE_weight = torch.tensor(wCE_weight)
                wCE_weight = len(wCE_weight) * wCE_weight / sum(wCE_weight)
                criterion = nn.CrossEntropyLoss()
                criterion_summed = nn.CrossEntropyLoss(reduction='sum')
            if args.loss_type == "CDT":
                criterion = CDTLoss(delta_list, gamma=gamma, weight=None, reduction=None, device=device)
                criterion_summed = CDTLoss(delta_list, gamma=gamma, weight=None, reduction="sum", device=device)
            if args.loss_type == "LDT":
                criterion = LDTLoss(delta_list, gamma=gamma, weight=None, reduction=None, device=device)
                criterion_summed = LDTLoss(delta_list, gamma=gamma, weight=None, reduction="sum", device=device)
            
            criterion.to(device)
            criterion_summed.to(device)


            # ------- Optimizer ----------------------------------------------------------------------------------------------------
            optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_decay_epochs,
                                                        gamma=args.lr_decay)
            
            # ------- Data Loggers --------------------------------------------------------------------------------------------------
            logger_train = logger()
            logger_test = logger()

            # ------- Train ---------------------------------------------------------------------------------------------------------
            cur_epochs = []
            for epoch in range(1, args.epochs + 1):
                print("print epoch")
                torch.cuda.empty_cache()

                train(model, criterion, args, device, train_loader, optimizer, epoch)
                lr_scheduler.step()
                
                if epoch in log_epoch_list:
                    cur_epochs.append(epoch)
                    torch.cuda.empty_cache()
                    analysis(logger_train, model, criterion_summed, args, device, analysis_loader, classifier, features, epoch)
                    analysis(logger_test, model, criterion_summed, args, device, test_loader, classifier, features, epoch)

                    save_logger(logger_train, exp_save_path, "logger_train")
                    save_logger(logger_test, exp_save_path, "logger_test")

            os.makedirs(exp_complete_flag, exist_ok=True)


if __name__ == '__main__':
    main()