# imblearn

The `imblearn` package contains implementation of constrained $\psi$-learning methods to optimize false positive rate, false negative rate, recall, precision and $F_\beta$ score. The details of the algorithms can be found in

1. [Liu and Zhu (2022): *On the consistent estimation of optimal Receiver Operating Characteristic (ROC) curve*](https://openreview.net/pdf?id=Ijq1_a6DESm)
2. [Liu and Zhu (2022): *classification for imbalanced dataset*](https://openreview.net/pdf?id=Ijq1_a6DESm).


To install `l2path` from [github](http://github.com), type in R console
```R
devtools::install_github("renxiongliu/imblearn")
```
Note that the installation above requires using R package [devtools](https://CRAN.R-project.org/package=devtools)
(which can be installed using `install.packages("devtools")`).

Please check the accompanying [vignette](https://github.com/renxiongliu/l2path/blob/main/vignettes/vignette.pdf) on how to use the `imblearn` package.