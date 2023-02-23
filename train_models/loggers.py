import numpy as np
import pickle
from scipy.sparse.linalg import svds
import torch

class logger:
    
    def __init__(self):
        # Loss and Accuracy
        self.accuracy     = []
        self.loss         = []
        self.reg_loss     = []
        self.acc_perclass = []
        self.n_c          = []
        self.N            = []
        
        # NC1
        self.Sw_invSb     = []

        self.M = []
        self.W = []
        self.muG = []


def Update_Geometry_Prop(logger, args, loss, reg_loss, M, W, Sw, net_accuracy, per_class_acc, n_c, N):
    
    # Number datapoints
    logger.N = N
    logger.n_c = n_c

    # Loss 
    logger.loss.append(loss)
    logger.reg_loss.append(reg_loss)

    # Accuracies
    logger.accuracy.append(net_accuracy)
    logger.acc_perclass.append(per_class_acc)
    
    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / args.K

    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.K-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T
    logger.Sw_invSb.append(np.trace(Sw @ inv_Sb))
    # -------------------------------------------------------------------------------------------

    logger.M.append(M.cpu().detach().numpy())
    logger.W.append(W.cpu().detach().numpy())
    logger.muG.append(muG.cpu().detach().numpy())

    return

def save_logger(logger, dir_path, file_name):
    save_path = dir_path + "/" + file_name + '.pkl'
    f_test = open(save_path, "wb")
    pickle.dump(logger, f_test)
    f_test.close()