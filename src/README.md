<h1 align="center"> An Ensemble Learning Approach with Gradient
Resampling for Class-imbalanced problems </h1>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/ChuangZhao-BCWF-orange">
  <img src="https://img.shields.io/github/stars/ChuangZhao/JOC">
  <img src="https://img.shields.io/github/forks/ChuangZhao/JOC">
  <img src="https://img.shields.io/github/issues/ChuangZhao/JOC">  
  <img src="https://img.shields.io/github/license/ChuangZhao/JOC"> -->
  <a href="https://github.com/Data-Designer/JOC">
    <img src="https://img.shields.io/badge/ChuangZhao-JOC-orange">
  </a>
  <a href="https://github.com/Data-Designer/JOC/stargazers">
    <img src="https://img.shields.io/github/stars/Data-Designer/JOC">
  </a>
  <a href="https://github.com/Data-Designer/JOC/network/members">
    <img src="https://img.shields.io/github/forks/Data-Designer/JOC">
  </a>
  <a href="https://github.com/Data-Designer/JOC/issues">
    <img src="https://img.shields.io/github/issues/Data-Designer/JOC">
  </a>
  <a href="https://github.com/Data-Designer/JOC/graphs/traffic">
    <img src="https://visitor-badge.glitch.me/badge?page_id=Data-Designer.JOC">
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
<a href="https://github.com/Data-Designer/JOC#contributors-"><img src="https://img.shields.io/badge/all_contributors-1-orange.svg"></a>
<!-- ALL-CONTRIBUTORS-BADGE:END -->
  <a href="https://github.com/Data-Designer/JOC/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Data-Designer/JOC">
  </a>
</p>

## About Our Work

In this paper, we propose a new approach from the sample-level classification difficulty identifying, sampling and ensemble learning. Accordingly, we design an ensemble approach in pipe with sample-level gradient resampling,  i.e., **Balanced Cascade with Filters (BCWF)**. Before that, as a preliminary exploration, we first design a **Hard Examples Mining Algorithm (HEM)** to explore the gradient distribution of classification difficulty of samples and identify the hard examples.

The figure below gives an overview of the our framework. 

![image-20220615204204109](https://s2.loli.net/2022/06/15/iFbzAw1R5ZWceJs.png)



## Requirements

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



## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<table>
  <tr>
    <td align="center"><a href="https://data-designer.github.io/"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Chuang Zhao</b></sub></a><br /><a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Data-Designer/JOC/commits?author=Data-Designer" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/ohmymamamiya"><img src="https://avatars.githubusercontent.com/u/26108487?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nanlin Liu</b></sub></a><br /><a href="#ideas-ZhiningLiu1998" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/ohmymamamiya" title="Code">💻</a></td>
  </tr>
</table>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!