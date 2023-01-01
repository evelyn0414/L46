# L46

Set up the environment for using the dataset:

```
cd FLamby
make install
conda activate flamby
```

If (like in Windows), the make command does not work (as in acs-gpu ended up with the following error)
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

To download the datasets:

Fed-Heart-Disease
```
cd FLamby
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Fed-ISIC2019
```
cd FLamby
cd flamby/datasets/fed_isic2019/dataset_creation_scripts
python download_isic.py --output-folder ./isic2019_dataset
python resize_images.py
```

Code for testing the dataset is in `main.py` and code for comparing the strategies is in `PersonalizedFL/my_main.py`.
