# Implicit_Geom_CE_Param_Imbalance_Data
A Repository to recreate the results of <a href="doc:introduction" target="_blank">"On the Implicit Geometry of Cross-Entropy Parameterizations for Label-Imbalanced Data" </a>.

# Reproduce Experimental Vs Theory Comparison
![plot](./figs/CDT_R_10_merged.png)
![plot](./figs/LDT_R_10_merged.png)

All scripts required to reproduce theory-vs-exp are provided in train_models. As an example, in order to produce the above results for CDT: \\

**CIFAR10 + ResNet18**
```bash
python main_deepnet.py --gpu --loss_type CDT --model ResNet18 --dataset CIFAR10
```

**MLP + ResNet18**
```bash
python main_deepnet.py --gpu --loss_type CDT --model ResNet18 --dataset CIFAR10
```

**UFM**
```bash
python main_UFM.py --loss_type CDT
```

The above commands will perform the experiments along a range of $\gamma \in \{-1.5, -1.25, ..., -0.25, 0.0, ..., 1.5 \}$ for $ R = 10 $ step imbalance ratio for one iteration (without data Augmentation). MNIST and CIFAR10 datasets will be downloaded into ```.\data ``` folder.  Results will be saved into proper directories in ```.\saved_logs ``` as ``` log.pkl ``` files.

In order to produce the plots, run :
```bash
python geom_compare.py --loss_type CDT
```

Same resutls can be reproduced for LDT.