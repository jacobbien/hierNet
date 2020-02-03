\name{hierNet.cv}
\alias{hierNet.cv}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Cross-validation function for hierNet}
\description{
Uses cross-validation to estimate the regularization parameter for hierNet
}
\usage{
hierNet.cv(fit, x, y, nfolds=10,folds=NULL,trace=0)
}
\arguments{
  \item{fit}{Object returned from call to hierNet.path or hierNet.logistic.path.  All parameter settings
will be taken from this object.}
  \item{x}{A matrix of predictors, where the rows are the samples and
    the columns are the predictors}
  \item{y}{A vector of observations, where length(y)  equals
    nrow(x)}
\item{nfolds}{Number of cross-validation folds}
\item{folds}{(Optional) user-supplied cross-validation folds.  If provided, nfolds is ignored.}
\item{trace}{Verbose output? 0=no, 1=yes}
}
\value{
\item{lamlist}{Vector of lambda values tried}
\item{cv.err}{Estimate of cross-validation error}
\item{cv.se}{Estimated  standard error of cross-validation estimate }
\item{lamhat}{lambda value minimizing cv.err}
\item{lamhat.1se}{largest lambda value with cv.err less than  or equal to min(cv.err)+ SE }
\item{folds}{Indices of folds used in cross-validation}
\item{yhat}{n by nlam matrix of predicted values.  Here, ith prediction
  is based on training on all folds that do not include the ith data
  point.}
\item{nonzero}{Vector giving number of non-zero coefficients for each
  lambda value}
\item{call}{The call to hierNet.cv}
}
\references{Bien, J., Taylor, J., Tibshirani, R., (2013) "A Lasso for Hierarchical Interactions." Annals of Statistics. 41(3). 1111-1141.}
\author{Jacob Bien and Robert Tibshirani}
\seealso{\link{hierNet},\link{hierNet.path}, \link{hierNet.logistic},\link{hierNet.logistic.path}}
\examples{
set.seed(12)
x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
fit=hierNet.path(x,y)
fitcv=hierNet.cv(fit,x,y)
print(fitcv)
plot(fitcv)


x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
y=1*(y>0)
fit=hierNet.logistic.path(x,y)
fitcv=hierNet.cv(fit,x,y)
print(fitcv)
plot(fitcv)
}
