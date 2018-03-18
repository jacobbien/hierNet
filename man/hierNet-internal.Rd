\name{hierNet-internal}
\title{Internal hierNet functions}
\alias{print.hierNet}
\alias{print.hierNet.path}
\alias{print.hierNet.cv}
\alias{plot.hierNet.cv}
\alias{critf.logistic}
\alias{Objective}
\alias{Objective.logistic}
\alias{compute.interactions.c}
\description{Internal hierNet functions}
\usage{
\method{print}{hierNet}(x,...)
\method{print}{hierNet.path}(x,...)
\method{print}{hierNet.cv}(x,...)
\method{plot}{hierNet.cv}(x,...)
critf.logistic(x, y, lam.l1, lam.l2, b0, bp, bn, th)
Objective(aa, x, y, lam.l1, lam.l2, xnum = NULL, zz = NULL, strong = TRUE, 
    sym.eps = 0.001)
Objective.logistic(aa, x, y, lam.l1, lam.l2, xnum = NULL, zz = NULL, strong = TRUE, 
    sym.eps = 0.001) 
compute.interactions.c(x, diagonal = TRUE)
}
\author{Jacob Bien and Robert Tibshirani}
\details{These  are internal functions,  not to be called by the user.}
\keyword{internal}
