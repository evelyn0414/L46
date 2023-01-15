# L46 Project Planning

## Goal

Evaluate multiple personalized FL strategies on real-world healthcare datasets and implement them in two frameworks.

### Breaking down:

- Understand the strategies
- Figure out the machine to run the project on => Cambridge CSD3
- Implement them and run the experiments
- Analyze the results and write the report

## Schedule

### Dec 2 - Dec 13

- Understand basic concepts
- Try out FLamby dataset suits
- Look into Flower and MsPFL source code
- Create Github repo

### Dec 14 - Dec 20

- Investigate HPC computing cluster
- Implement a basic version using Fedavg and Fed-Heart-Disease locally

### Dec 21 - Jan 3

- Implement FedProx and FedBN in Flower
- Run Experiments using FedAvg and FedBN on all three datasets
- Look into the implementation of FedAP and MetaFed
- Run MetaFed in MsPFL

### Jan 4 - Jan 10

- Start writing the report
- Debug with the `getallfea` function for FedAP and `get_sel_fea` function for MetaFed
- Run experiments for FedAP on both frameworks

### Jan 11 - Jan 17

- Organize and analyze all experimental results
- Work on the report

## Decision-making

Originally, we came up the idea of testing pFL strategies on real-world healthcare dataset suite FLamby and implementing them in the Flower framework. At first we chose Fed-Camelyon16, but it is too big and require too much computing resources, so afterwards we switched to Fed-IXI instead.

We then came across MsPFL framework which lead us to choose these strategies supported by it, and compare them across the two frameworks (MetaFed is run but not included since we do not have enough time to implement cyclic training in Flower).

When running the experiments for MsPFL, we spent much time debugging with the HPC environment and FedAP implementation for the new models. The original models supported in MsPFL are straightforward in design, and we ended up implementing a dfs method that is applicable to most models. However, after switching the dataset and task, we had to write a customized traversal function for 3D UNet since it is not simply sequential.

For the experiments in Flower, the main problem is the SSD memory and CUDA out-of-memory issues with Ray, due to the large size of our Fed-ISIC2019 dataset. We modify and optimize both our main file and Flower source code to deal with this issue.

During this project, we get deeper understanding of FL and the strategies, as well as architecture of the deep learning models. We hope to further extend this work in the future, by including more strategies and generating more insights about their suitability to different data conditions, as well as contributing to the source code of the frameworks.
