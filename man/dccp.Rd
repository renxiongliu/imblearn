% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DCCP.R
\name{dccp}
\alias{dccp}
\title{Core algorithm for improving a linear classifier in terms of some metrics}
\usage{
dccp(
  X,
  Y,
  beta_init,
  b_init,
  gamma = 0.001,
  psi_k = 10,
  max_iter_num = 25,
  max_rel_gap = 5 * 0.001,
  metrics = "ROC"
)
}
\arguments{
\item{X}{A \eqn{n \times p} feature matrix.}

\item{Y}{A \eqn{n \times 1} binary vector with value -1 and +1.}

\item{beta_init}{Slope coefficient for initialized discriminant function.}

\item{b_init}{Intercept coefficient for initialized discriminant function.}

\item{gamma}{hyperparameter value of regularizer in the object function.}

\item{psi_k}{the hyperparameter in psi function: psi(x, psi_k) = min(1, max(0, 1-k*x))}

\item{max_iter_num}{maximal number of iterations in DCCP iterations.}

\item{max_rel_gap}{relative tolerance of object function value improvement during iteration.}

\item{metrics}{Specify the metric to improve for initialized linear classifier
"ROC" means optimizing type I/II error while "PR" means optimize
recall/precision.}
}
\value{
beta: Generated slope coefficient

b: Generated intercept coefficient
}
\description{
Compute improved linear classifier to initialized linear classifier under certain metrics.
}
