# MsPFL implementation

## Introduction and code navigation

The framework implementation is forked from [microsoft/PersonalizedFL](https://github.com/microsoft/PersonalizedFL.git).

Our experiment code can be run by simply executing our own main python file,

````bash
python my_main.py --alg fedavg
````

with only the strategy to be parsed and other parameters defined in the code.

To switch between datasets, only the import statements and definitions in the beginning need to be changed (see the annotation).

We run experiments of all five strategies supported by the framework. 

1. FedAvg [1].
2. FedProx [2].
3. FedBN [3].
4. FedAP [4].
5. MetaFed [5].

The results of MetaFed is not compared in the report as it is not implemented in the Flower framework. We still include the results here.

The trained models and training logs of server&client performance in each round are all stored in `./cks/` except the models for `Fed-ISIC2019` which is too large to be pushed to GitHub. The results are concluded in tables in `res.csv`.

To understand the code: The algorithm classes are defined in separate python files in `./alg/`, and we have made modification to each one. The communication function is defined in `./alg/core/comm.py`, which is the original implement.

Our change to the models to accommodate fedAP and metaFed strategies was made to the `FLamby` codebase, which can be found in `../FLamby/flamby/datasets/DATASET_NAME/model.py`. We define a `getallfea` and `get_sel_fea` for each of the three models.

## Results

The results of MetaFed is not compared in the report as it is not implemented in the Flower framework. We still include the results here.

In Fed-Heart-Disease with BN, MetaFed comes first in each client's model performance and achieves the best fairness measured by variance. However, in the other two datasets, MetaFed ends up with the worst performance and worst fairness. The experiments cannot be repeated in Flower and we have not yet done more experiments to analyze the reason behind.

#### **Fed-Heart-Disease (1-layer MLP)**

|             | server | 0      | 1      | 2      | 3      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7717 | 0.7596 | 0.7640 | 0.7500 | 0.6444 | 0.7295     | 0.057      |
| **fedProx** | 0.7677 | 0.7500 | 0.7528 | 0.8125 | 0.6444 | 0.7399     | 0.070      |
| **fedbn**   | /      | / | / | / | / | /     | /      |
| **fedAP** \*  | /      | 0.7596 | 0.7341 | 0.8125 | 0.7333 | 0.7599     | 0.037      |
| **metaFed** | /      | 0.7756 | 0.7640 | 0.7500 | 0.7111 | 0.7502     | 0.028      |

(\* weight matrix here is calculated with output of the fully connected layer)

#### Fed-Heart-Disease (2-layer MLP + BN)

|             | server | 0      | 1      | 2      | 3      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7087 | 0.6699 | 0.5918 | 0.9167 | 0.8370 | 0.7538     | 0.149      |
| **fedProx** | 0.6877 | 0.6603 | 0.6180 | 0.8958 | 0.7852 | 0.7398     | 0.126      |
| **fedbn**   | /      | 0.7212 | 0.7865 | 0.9375 | 0.7556 | 0.8002     | 0.095      |
| **fedAP**   | /      | 0.7212 | 0.7865 | 0.9375 | 0.7778 | 0.8057     | 0.092      |
| **metaFed** | /      | 0.7660 | 0.7865 | 0.9375 | 0.8074 | 0.8244     | 0.077      |

#### Fed-IXI

|             | server | 0      | 1      | 2      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.9890 | 0.9892 | 0.9898 | 0.9869 | 0.9886     | 0.002      |
| **fedProx** | 0.9882 | 0.9881 | 0.9889 | 0.9868 | 0.9879     | 0.001      |
| **fedbn**   | /      | 0.9891 | 0.9896 | 0.9871 | 0.9886     | 0.001      |
| **fedAP**   | /      | 0.9892 | 0.9896 | 0.9874 | 0.9888     | 0.001      |
| **metaFed** | /      | 0.9825 | 0.9819 | 0.9741 | 0.9795     | 0.005      |

#### Fed-ISIC2019

|             | server | 0      | 1      | 2      | 3      | 4      | 5      | client avg | client std |
| ----------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ---------- | ---------- |
| **fedAvg**  | 0.7270 | 0.7913 | 0.7116 | 0.6486 | 0.5470 | 0.4587 | 0.7335 | 0.6484     | 0.1248     |
| **fedProx** | 0.7081 | 0.7638 | 0.6694 | 0.6515 | 0.5364 | 0.4776 | 0.7646 | 0.6439     | 0.1174     |
| **fedbn**   | /      | 0.7746 | 0.5863 | 0.7389 | 0.7067 | 0.6879 | 0.7399 | 0.7057     | 0.0657     |
| **fedAP**   | /      | 0.8261 | 0.7160 | 0.8162 | 0.7340 | 0.6925 | 0.7515 | 0.7560     | 0.0542     |
| **metaFed** | /      | 0.7964 | 0.7263 | 0.7072 | 0.4886 | 0.4380 | 0.6772 | 0.6390     | 0.1425     |

## Reference

[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[3] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[4] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[5] Yiqiang, Chen, et al. "MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare." FL-IJCAI Workshop 2022.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
