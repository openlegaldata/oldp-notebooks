# Jupyter Notebooks for the OLDP Project

This repository contains serveral Jupyter Notebooks that demonstrate how to use data from the [Open Legal Data](https://openlegaldata.io/) project.

Python Examples:

- How to use the OLDP Python client: `client_demo`

- Train a N-gram model based on case text: `ngram_model`

- Use extract and add annotations (litigation values) to cases: `litigtation_value`

- Visualization data from the citation/references dump: `references`


## Dependencies

Get a copy of the repository:
```
$ git clone git@github.com:openlegaldata/oldp-notebooks.git
```

Before getting started you have to install the Python dependencies. The commands assume that you use `pipenv`:
```
$ cd oldp-notebooks
$ pipenv --python 3.7
$ pipenv install
```

Each notebook may have its individual dependencies. Please check the introduction inside the notebooks.