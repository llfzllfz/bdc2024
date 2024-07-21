# file structure

```
|-project
	|-README.md
	|- ...
|-README.md
|-cenn
|-train
|-global
|-split_data_train.ipynb # the file to split the global and generate the cenn and the train, the train fold for train, the cenn fold for test
```

# Usage

First, download the dataset with the global fold, and run the split_data_train.ipynb file to generate the train, cenn fold.

Then, use 

```
cd project
```

and run temp with 

```
python train.py --train temp --test all
```

or run wind with 

```
python train.py --train wind --test all
```

Also, you can change some parameters with config.json. The detailed information is in project/README.md.

