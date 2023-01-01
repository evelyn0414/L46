# L46

Set up the environment for using the dataset:

```
cd FLamby
make install
conda activate flamby
```

If the make command does not work (as in acs-gpu ended up with the following error)
```
/bin/sh: 2: /opt/conda/envs/flamby/bin/pip: not found
make: *** [Makefile:8: install] Error 127
```

try
```
pip install -e .[all_extra]
```

To download new datasets:

```
cd FLamby
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Code for testing the dataset is in `main.py` and code for comparing the strategies is in `PersonalizedFL/my_main.py`.
