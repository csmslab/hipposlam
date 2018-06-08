# hippoSLAM
a Simultaneous Localization And Mapping algorithm based on mammal (particularly
rodent) _hippocampal_ navigation paradigms.

View/run the notebooks here:

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/acellon/hipposlam/master?filepath=biocas_notebook.ipynb)

This repo includes:

 File | Description
----|---------
`vco.py` | a Python library defining model classes and helpful functions for computation and plotting.
`vco_test.ipynb` | a Jupyter notebook demonstrating `vco.py`.
`biocas_notebook.ipynb` | Jupyter notebook describing important math and simulation details for our BioCAS 2018 paper.
`de_almeida.ipynb` | Jupyter notebook demonstrating [de Almeida _et al._, 2009](http://www.jneurosci.org/content/29/23/7504.long).
`scores.ipynb` | Jupyter notebook developing/discussing various scoring concepts for navigational cells.

Still under construction:

File | Description
---|---
`hc_nav.ipynb` | a Jupyter notebook demonstrating [Welday _et al._, 2011](http://www.jneurosci.org/content/31/45/16157.long).
`solstad.ipynb` | Jupyter notebook demonstrating [Solstad _et al._, 2006](https://www.ncbi.nlm.nih.gov/pubmed/17094145).
`place_from_grid.ipynb` | Jupyter notebook testing a few ways to hierarchically build up place cells from grid cells.
