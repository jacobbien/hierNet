\name{hierNet.logistic.path}
\alias{hierNet.logistic.path}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Fit a path of logistic hierNet models- lasso models with interactions}
\description{
One of the main functions in the hierNet package.
Fits a logistic path of hierNet models over different values of the regularization parameter.
Calls hierNet.logistic, which builds a regression model with hierarchically constrained pairwise interactions.
 Required inputs are an x matrix of
features (the columns are the features) and a y vector of
values.
Reasonably fast for moderate sized problems (100-200 variables). We are currently working
on a alternate algorithm for large scale problems.

}
\usage{
hierNet.logistic.path(x, y,
           lamlist = NULL, delta=1e-8, minlam = NULL, maxlam = NULL, flmin=.01, nlam = 20,
           diagonal = TRUE, strong = FALSE, aa = NULL, zz = NULL,
           stand.main = TRUE, stand.int = FALSE,
           rho = nrow(x), niter = 100, sym.eps = 0.001, 
           step = 1, maxiter = 2000, backtrack = 0.2, tol = 1e-05, trace = 0)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{A matrix of predictors, where the rows are the samples and
    the columns are the predictors}
  \item{y}{A vector of observations equal to 0 or 1, where length(y)  equals
    nrow(x)}
  \item{lamlist}{Optional vector of values of lambda (the regularization parameter).  L1 penalty param is \code{lambda * (1-delta)}.}
  \item{delta}{Elastic Net parameter. Squared L2 penalty param is \code{lambda * delta}. Not a tuning parameter: Think of as fixed and small. Default 1e-8.}
\item{minlam}{Optional minimum value for  lambda} 
\item{maxlam}{Optional maximum value for  lambda}
\item{flmin}{Fraction of maxlam; minlam= flmin*maxlam. If computation is slow, try increasing
flmin to focus on the sparser part of the path}
\item{nlam}{Number of values of lambda to be tried}
\item{diagonal}{Flag specifying whether to include "pure" quadratic
  terms, th_{jj}X_j^2, in the model.  Default TRUE.}
\item{stand.main}{Should main effects be standardized? Default TRUE}
\item{stand.int}{Should interactions be standardized? Default FALSE}
  \item{strong}{Flag specifying strong hierarchy (TRUE) or weak hierarchy (FALSE). Default FALSE}
  \item{aa}{An *optional* argument, a list with results from a previous call}
\item{zz}{An  *optional* argument, a matrix whose columns are products of features, computed by the function compute.interactions.c}
  \item{rho}{ADMM parameter: tuning parameter (>0) for ADMM. If there are convergence
    problems, try decreasing \code{rho}. Default n.}
  \item{niter}{ADMM parameter: number of iterations}
\item{sym.eps}{ADMM parameter Thresholding for symmetrizing with strong=TRUE}
\item{step}{Stepsize for generalized gradient descent}
\item{maxiter}{Maximum number of iterations for generalized gradient descent}
\item{backtrack}{Backtrack parameter for generalized gradient descent}
\item{tol}{Error tolerance parameter for generalized gradient descent}
\item{trace}{Output option; trace=1 gives verbose output}
}
\value{
\item{bp}{p by nlam  matrix of estimated  "positive part"  main
  effects (p=#features)}
\item{bn}{p by nlam  matrix of estimated  "negative part"  main effects}
\item{th}{p by p by nlam array of estimated interaction coefficients}
\item{obj}{nlam values of objective function, one per lambda value}
\item{lamlist}{Vector of values of lambda used}
\item{mx}{p-vector of  column means of x}
\item{sx}{p-vector of  column standard deviations of x}
\item{my}{mean of y}
\item{mzz}{column means of feature product matrix}
\item{szz}{column standard deviations of feature product matrix}
}
\references{Bien, J., Taylor, J., Tibshirani, R., (2013) "A Lasso for Hierarchical Interactions." Annals of Statistics. 41(3). 1111-1141.}
\author{Jacob Bien and Robert Tibshirani}
\seealso{\link{hierNet},\link{predict.hierNet}, \link{hierNet.cv}}
\examples{
set.seed(12)
x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
y=1*(y>0)
fit=hierNet.logistic.path(x,y)
print(fit)
}
