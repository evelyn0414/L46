# L46

environment for using the dataset:

```
cd FLamby
make install
conda activate flamby
```

downloading new datasets:

```
cd FLamby
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

code for testing the dataset is in `main.py` and code for comparing the strategies is in `PersonalizedFL/my_main.py`.
