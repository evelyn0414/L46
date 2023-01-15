# L46 project

This is the codebase for the L46 project *Personalized Federated Learning in Real-world Healthcare Applications across Different Frameworks*.

## Dataset

### Environment

Set up the environment for using the dataset:

```
cd FLamby
make install
conda activate flamby
```

If the make command does not work ((like in Windows, or as in acs-gpu, ended up with the following error)
```python resize_images.py
/bin/sh: 2: /opt/conda/envs/flamby/bin/pip: not found
make: *** [Makefile:8: install] Error 127
```

Try this instead:
```
cd FLamby
conda env create -f environment.yml
conda activate flamby
pip install -e .[all_extra]
```

In HPC we also need to run the followinig before `conda activate`
```
conda init bash
source ~/.bash_profile
```

Another useful command to debug with HPC environment when encountering error with PyTorch:

```
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

### Downloading

The command line to download the datasets:

Fed-Heart-Disease
```
cd FLamby/flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Fed-IXI

```
cd FLamby/flamby/datasets/fed_ixi/dataset_creation_scripts
python download.py -o IXI-Dataset
```

Fed-ISIC2019

```
cd FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts
python download_isic.py --output-folder ./isic2019_dataset
python resize_images.py
```

## MsPFL implementation

The main function for comparing the strategies using MsPFL framework is in `MsPFL/my_main.py`,  and concluded experimental results can be found in `MsPFL/res.csv`. We have another `README.md` file in the folder to help execute and navigate the code in detail.

## Flwr implementation

We further provide instructions in the folder.
