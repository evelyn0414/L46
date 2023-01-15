# Flower implementation

## Introduction

Our main files use the code of [L46 lab 3](https://colab.research.google.com/drive/1CqzPG4r0qWcIuhPlPtkwGL3vKAKjJQRD?usp=sharing) as a starting point. Lots of modifications are made in both the main files and Flower source code. Please refer to our paper for details.

## How to setup and run

Environment setup:
```
git clone https://github.com/owkin/FLamby Flamby
cd Flamby (put L46_code.ipynb / L46_code.py in this folder)
make install
conda activate flamby
pip install torchmetrics
pip install tqdm
pip install torchsummary
pip install -U flwr["simulation"]
```

Only for running L46_code.ipy in local PC:
```
conda install jupyter
conda install -n flamby nb_conda_kernels
```

How to modify Flower source code:
```
conda activate flamby
pip show flwr
open /***/***/anaconda3/envs/flamby/lib/python3.8/site-packages/flwr in vscode
```

Note:
- We git clone Flamby only to use its benchmarks, dataloaders, and utility functions.
- L46_code.py / L46_code.ipynb are the main files. They have the same code.
- flwr is the modified source code of Flower.

## Results

Fed-Heart-Disease (1-layer MLP)

|             | server | 0      | 1      | 2      | 3      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7848 | 0.7820 | 0.6966 | 0.7500 | 0.6889 | 0.7294     | 0.044      |
| **fedProx** | 0.7756 | 0.7885 | 0.7303 | 0.7500 | 0.6889 | 0.7394     | 0.041      |
| **fedbn**   | /      | /      | /      | /      | /      | /          | /          |
| **fedAP**   | /      | /      | /      | /      | /      | /          | /          |

Fed-Heart-Disease (2-layer MLP + BN)

|             | server | 0      | 1      | 2      | 3      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7175 | 0.6795 | 0.5917 | 0.8750 | 0.7955 | 0.7354     | 0.125      |
| **fedProx** | 0.7149 | 0.6635 | 0.5841 | 0.8125 | 0.7727 | 0.7082     | 0.104      |
| **fedbn**   | /      | 0.6827 | 0.7219 | 0.9375 | 0.8182 | 0.7901     | 0.114      |
| **fedAP**   | /      | 0.6859 | 0.7636 | 0.9375 | 0.7879 | 0.7937     | 0.105      |

Fed-IXI

|             | server | 0      | 1      | 2      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.9840 | 0.9836 | 0.9858 | 0.9838 | 0.9844     | 0.001      |
| **fedProx** | 0.9828 | 0.9840 | 0.9848 | 0.9832 | 0.9840     | 0.001      |
| **fedbn**   | /      | 0.9841 | 0.9855 | 0.9840 | 0.9845     | 0.001      |
| **fedAP**   | /      | 0.9850 | 0.9857 | 0.9841 | 0.9849     | 0.001      |

Fed-ISIC2019

|             | server | 0      | 1      | 2      | 3      | 4      | 5      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7405 | 0.7735 | 0.6768 | 0.6315 | 0.5236 | 0.4646 | 0.7434 | 0.6356     | 0.1218     |
| **fedProx** | 0.7335 | 0.7553 | 0.6281 | 0.6429 | 0.5392 | 0.5058 | 0.7738 | 0.6409     | 0.1091     |
| **fedbn**   | /      | 0.7747 | 0.6197 | 0.7415 | 0.7339 | 0.6818 | 0.7316 | 0.7139     | 0.0549     |
| **fedAP**   | /      | 0.8380 | 0.6924 | 0.7981 | 0.7406 | 0.6944 | 0.7462 | 0.7516     | 0.0575     |

## References

[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[3] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[4] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[5] Beutel, Daniel J., et al. "Flower: A Friendly Federated Learning Research Framework." ArXiv, abs/2007.14390.

## Acknowledgements

We thank for [Flower](https://flower.dev/) for providing the excellent federated learning framework.
