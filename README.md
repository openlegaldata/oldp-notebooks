# Jupyter Notebooks for the OLDP Project

This repository contains serveral Jupyter Notebooks that demonstrate how to use data from the [Open Legal Data](https://openlegaldata.io/) project.

Python Examples:

- How to use the OLDP Python client: `client_demo`

- Train a N-gram model based on case text: `ngram_model`

- Use extract and add annotations (litigation values) to cases: `litigtation_value`

- Visualization data from the citation/references dump: `references`

- Publication statistics: `publication-stats`

- Topic modeling and topics over time: `topics`


## Getting started

### Installation

Get a copy of the repository:
```bash
$ git clone git@github.com:openlegaldata/oldp-notebooks.git
```

Before getting started you have to install the Python dependencies. We recommend using Python version >= 3.12. The commands assume that you use `pip` or `uv`:
```bash
$ cd oldp-notebooks

# create virtual environment
# ... with venv
$ python3 -m venv .venv

# ... or with uv
$  uv venv --python=3.12

# activate environment
source .venv/bin/activate

# install depdencies
# ... with pip
$ pip install -r requirements.txt


# ... with uv
$ uv pip install -r requirements.txt
```

Each notebook may have its individual dependencies. Please check the introduction inside the notebooks.

### Run notebooks

To run the available notebooks via Jupyter you need to run the following command:

```bash
jupyter notebook
```

## Google Colab

You can also run the Jupyter notebooks directly on Google Colab:

- [Publication Statistics](https://colab.research.google.com/github/openlegaldata/oldp-notebooks/blob/master/notebooks/publication-stats.ipynb)


## Citation

Please cite the following [research paper](https://arxiv.org/abs/2005.13342), if you use our code or data:

```bibtex
@inproceedings{10.1145/3383583.3398616,
author = {Ostendorff, Malte and Blume, Till and Ostendorff, Saskia},
title = {Towards an Open Platform for Legal Information},
year = {2020},
isbn = {9781450375856},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3383583.3398616},
doi = {10.1145/3383583.3398616},
booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries in 2020},
pages = {385–388},
numpages = {4},
keywords = {open data, open source, legal information system, legal data},
location = {Virtual Event, China},
series = {JCDL '20}
}
```

## License 

MIT