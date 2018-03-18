\name{hierNet.logistic}
\alias{hierNet.logistic}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{A logistic regression Lasso for interactions}
\description{
One of the main functions in the hierNet package.
Builds a logistic regression model with hierarchically constrained pairwise interactions.
 Required inputs are an x matrix of
features (the columns are the features) and a y vector of
values.
Reasonably fast for moderate sized problems (100-200 variables). We are currently working
on a alternate algorithm for large scale problems.

}
\usage{
hierNet.logistic(x, y, lam, delta=1e-8, diagonal=TRUE, strong=FALSE, aa=NULL, zz=NULL,
                 center=TRUE, stand.main=TRUE, stand.int=FALSE,
                 rho=nrow(x), niter=100, sym.eps=1e-3,# ADMM params
                 step=1, maxiter=2000, backtrack=0.2, tol=1e-5, trace=1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{A matrix of predictors, where the rows are the samples and
    the columns are the predictors}
  \item{y}{A vector of observations, with values 0 or 1, where length(y)  equals
    nrow(x)}
  \item{lam}{Regularization parameter (>0).  L1 penalty param is \code{lam * (1-delta)}.}
  \item{delta}{Elastic Net parameter. Squared L2 penalty param is \code{lam * delta}. Not a tuning parameter: Think of as fixed and small. Default 1e-8.}
  \item{diagonal}{Flag specifying whether to include "pure" quadratic
    terms, th_{jj}X_j^2, in the model.  Default TRUE.}  
  \item{strong}{Flag specifying strong hierarchy (TRUE) or weak hierarchy (FALSE). Default FALSE}
  \item{aa}{An *optional* argument, a list with results from a previous call}
  \item{zz}{An  *optional* argument, a matrix whose columns are products of features, computed by the function compute.interactions.c}
  \item{center}{Should features be centered? Default TRUE; FALSE should rarely be used.  This option is available for special uses only}
  \item{stand.main}{Should main effects be standardized? Default TRUE}
  \item{stand.int}{Should interactions be standardized? Default FALSE}
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
  \item{b0}{Intercept}
  \item{bp}{p-vector of estimated  "positive part"  main effect (p=#features)}
  \item{bn}{p-vector of estimated  "negative part"  main effect; overall main effect estimated coefficients are bp-bn }
  \item{th}{Matrix of estimated interaction coefficients, of dimension p by p}
  \item{obj}{Value of objective function at minimum.}
  \item{lam}{Value of lambda used}
  \item{type}{Type of model fit- "gaussian" or "logistic" (binomial)}
  \item{mx}{p-vector of column means of x}
  \item{my}{Mean of y}
  \item{sx}{p-vector of  column standard deviations of x}
  \item{mzz}{ column means of feature product matrix}
  \item{call}{The call to hierNet}
}
\references{Bien, J., Taylor, J., Tibshirani, R., (2013) "A Lasso for Hierarchical Interactions." Annals of Statistics. 41(3). 1111-1141.}
\author{Jacob Bien and Robert Tibshirani}
\seealso{\link{predict.hierNet.logistic},link{hierNet.logistic.path}}
\examples{
set.seed(12)
x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
y=1*(y>0)
fit=hierNet.logistic(x,y,lam=5)
print(fit)
}
