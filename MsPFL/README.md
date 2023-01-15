# MsPFL implementation

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

The trained models and training logs of server&client performance in each round are all stored in `./cks/` except the models for `Fed-ISIC2019` which is too large to be pushed to GitHub. The results are concluded in tables in `res.csv`.

To understand the code: The algorithm classes are defined in separate python files in `./alg/`, and we have made modification to each one. The communication function is defined in `./alg/core/comm.py`, which is the original implement.

Our change to the models to accommodate fedAP and metaFed strategies was made to the `FLamby` codebase, which can be found in `../FLamby/flamby/datasets/DATASET_NAME/model.py`. We define a `getallfea` and `get_sel_fea` for each of the three models.

## Reference

[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[3] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[4] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[5] Yiqiang, Chen, et al. "MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare." FL-IJCAI Workshop 2022.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
