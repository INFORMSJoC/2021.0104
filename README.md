[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# An Ensemble Learning Approach with Gradient Resampling for Class-imbalanced problems

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[An Ensemble Learning Approach with Gradient Resampling for Class-imbalanced problems]() by Chuang Zhao. 
The snapshot is based on 
[JOC](https://github.com/Data-Designer/JOC) 
in the development repository. 

**Important: This code is being developed on an on-going basis at 
https://github.com/Data-Designer/JOC. Please go there if you would like to
get a more recent version or would like support**

## Cite

[![DOI](https://zenodo.org/badge/569335708.svg)](https://zenodo.org/badge/latestdoi/569335708)

elow is the BibTex for citing this data.

```
@article{PutABibTexKeyHere,
  author =        {Hongke Zhao, Chuang Zhao, Xi Zhang, Nanlin Liu, Hengshu Zhu, Qi Liu, Hui Xiong},
  publisher =     {INFORMS Journal on Computing},
  title =         {An Ensemble Learning Approach with Gradient Resampling for Class-imbalanced problems v2021.0104},
  year =          {2022},
  doi =           {10.5281/zenodo.6360996},
  url =           {https://github.com/INFORMSJoC/2021.0104},
}  
```
## Description

The goal of this software is to demonstrate the effect of *An Ensemble Learning Approach with Gradient Resampling for Class-imbalanced problems* optimization.

In this paper, we propose a new approach from the sample-level classification difficulty identifying, sampling and ensemble learning. Accordingly, we design an ensemble approach in pipe with sample-level gradient resampling,  i.e., **Balanced Cascade with Filters (BCWF)**. Before that, as a preliminary exploration, we first design a **Hard Examples Mining Algorithm (HEM)** to explore the gradient distribution of classification difficulty of samples and identify the hard examples.

The figure below gives an overview of the our framework. 

![image-20220615204204109](https://s2.loli.net/2022/06/15/iFbzAw1R5ZWceJs.png)

## Building

**Main dependencies:**

- [Python](https://www.python.org/) (>=3.5)
- [pandas](https://pandas.pydata.org/) (>=0.23.4)
- [numpy](https://numpy.org/) (>=1.11)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.20.1)
- [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/index.html) (=0.5.0, optional, for baseline methods)

To install requirements, run:

```Shell
pip install -r requirements.txt
```

## Usage

A typical usage example:

```python
# Define model
model_class = BcwfH(dataset_name, T=15) 
# Model calculation
model_class.apply_all()
# Metrics
metrics = model_class.display()
```

You can run .py file too.

Here is an example:

```powershell
# Run Model
python main.py
# Hyper-test
python hyper.py
# Get comparsion
python result.py
```

## Result

![image-20221119101216770](https://s2.loli.net/2022/11/19/wtGTrxNMOqlRKEz.png)

![image-20221119101233632](https://s2.loli.net/2022/11/19/ldLYe9prM7sykaU.png)

![image-20221119101312282](https://s2.loli.net/2022/11/19/6pPxnDjbAkw5EmL.png)

## Replicating

To replicate the results in any of the tables in the paper, simply follow the Usage or refer to https://github.com/Data-Designer/JOC.
