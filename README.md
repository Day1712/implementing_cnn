# implementing_cnn

## Structure


```
root/
├── .gitignore             
├── data/                  # Download MNIST & Var-Res MNIST here
├── modules/               # Shared Code
│   ├── __init__.py
│   ├── convolution.py     # STUDENT 1: Custom Conv2DFunc & unfold (Q1-6)
│   ├── networks.py        # STUDENT 2 & 3: FixedCNN (Q7) & VarResNet (Q15)
│   └── data_loader.py     # STUDENT 3: Manual batching logic for Var-Res (Q14) 
├── scripts/               # Optional: 
│   ├── train_baseline.py  # Run the FixedCNN training (Student 2)
│   └── train_var_res.py   # Run the Variable Resolution training (Student 3)
└── notebooks/             # Individual Workspaces for Analysis/Plots
    ├── 1_gradients.ipynb  # STUDENT 1: Testing gradients & Unfold logic (Q1-6)
    ├── 2_baseline.ipynb   # STUDENT 2: FixedCNN experiments & Augmentation (Q7-9, 12-13)
    ├── 3_var_res.ipynb    # STUDENT 3: VarResNet vs Fixed comparison (Q14-17)
```