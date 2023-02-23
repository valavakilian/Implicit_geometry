import torch
from tqdm import tqdm
from scipy.sparse.linalg import svds
import numpy as np

from loggers import *

# ------- train fcn ---------------------------------------------------------------------------------------------------
def train(model, criterion, args, device, train_loader, optimizer, epoch):
    model.train()

    n_c = {}
    for c in range(0, args.K):
        n_c[c] = 0

    per_class_acc = {}
    for c in range(0, args.K):
        per_class_acc[c] = 0

    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != args.batch_size:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)

        predicted = torch.argmax(out, dim=1)
        loss.backward()
        optimizer.step()

        accuracy = torch.mean((predicted == target).float()).item()

        pbar.update(1)
        pbar.set_description(
            'Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t'
            'Batch Loss: {:.6f} \t'
            'Batch Accuracy: {:.6f}'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                accuracy))

        for c in range(0, args.K):
            n_c[c] += sum((target == c)).item()

        for c in range(0, args.K):
            per_class_acc[c] += ((predicted == target) * (target == c)).sum().item()

        if args.debug and batch_idx > 20:
            break
    pbar.close()

    for c in range(0, args.K):
        per_class_acc[c] /= n_c[c]

    return per_class_acc


# ------- analysis fcn ------------------------------------------------------------------------------------------------
def analysis(logger, model, criterion_summed, args, device, analysis_loader, classifier, features_hook, epoch):
    model.eval()

    N             = 0
    mean          = [0 for _ in range(args.K)]
    Sw            = 0
    Sw_C = [0 for _ in range(args.K)]

    loss          = 0
    net_accuracy   = 0

    n_c = {}
    per_class_acc = {}
    for c in range(0, args.K):
        n_c[c] = 0
        per_class_acc[c] = 0
    
    with torch.no_grad():
        for computation in ['Mean', 'Cov']:
            pbar = tqdm(total=len(analysis_loader), position=0, leave=True)
            for batch_idx, (data, target) in enumerate(analysis_loader, start=1):
                torch.cuda.empty_cache()
                
                data, target = data.to(device), target.to(device)

                output = model(data)
                h = features_hook.value.data.view(data.shape[0],-1) # B CHW

                predicted = torch.argmax(output, dim=1)                                    

                # during calculation of class means, calculate loss
                if computation == 'Mean':
                    loss += criterion_summed(output, target).item()

                for c in range(0, args.K):    

                    # features belonging to class c
                    idxs = (target == c).nonzero(as_tuple=True)[0]

                    # skip if no class-c in this batch
                    if len(idxs) == 0: 
                        continue

                    h_c = h[idxs,:].double() # B CHW

                    if computation == 'Mean':
                        # update class means
                        mean[c] += torch.sum(h_c, dim=0) # CHW
                        n_c[c] += h_c.shape[0]
                        N += h_c.shape[0]
                        # per class classifier accuracy
                        per_class_acc[c] += ((predicted == target) * (target == c)).sum().item()

                    elif computation == 'Cov':
                        # update within-class cov
                        z = h_c - mean[c].unsqueeze(0) # B CHW

                        # for loop - for solving memory issue :((
                        for z_i in range(z.shape[0]):
                            temp = torch.matmul(z[z_i, :].reshape((-1, 1)), z[z_i, :].reshape((1, -1)))
                            Sw += temp
                            Sw_C[c] += temp

                        # per class correct predictions
                        net_pred_for_c = torch.argmax(output[idxs,:], dim=1)
                        net_accuracy += (net_pred_for_c == target[idxs]).sum().item()

                if args.debug and batch_idx > 20:
                    break

                pbar.update(1)
                pbar.set_description(
                    'Analysis {}\t'
                    'Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        computation,
                        epoch,
                        batch_idx,
                        len(analysis_loader),
                        100. * batch_idx/ len(analysis_loader)))

                if args.debug and batch_idx > 20:
                    break
            pbar.close()

            if computation == 'Mean':
                for c in range(args.K):
                    mean[c] /= n_c[c]
                    M = torch.stack(mean).T
                loss /= N
            elif computation == 'Cov':
                Sw /= N
                for c in range(0, args.K):
                    Sw_C[c] /= n_c[c]
    
    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * args.weight_decay * torch.sum(param**2).item()

    net_accuracy = net_accuracy / N
    for c in range(0, args.K):
        per_class_acc[c] /= n_c[c]
    
    # avg norm
    W  = classifier.weight
    
    Update_Geometry_Prop(logger, args, loss, reg_loss, M, W, Sw, net_accuracy, per_class_acc, n_c, N)

    return