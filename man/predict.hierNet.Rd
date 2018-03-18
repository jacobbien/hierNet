\name{predict.hierNet}
\alias{predict.hierNet}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Prediction function for hierNet and hierNet.logistic.}
\description{
A function to perform prediction, using an x matrix and the output of
the "hierNet" or "hiernet.logistic" function.
}
\usage{
\method{predict}{hierNet}(object, newx, newzz=NULL, ...)
}
\arguments{
  \item{object}{The results of a call to the "hierNet" or "hierNet.path" or
 function. The
    coefficients that are part of this object will be used for
    making predictions.}
  \item{newx}{The new x at which predictions should be made. Can be a
    vector 
     or a matrix  (one obseration per row).}
  \item{newzz}{Optional matrix of products of columns of newx, computed by compute.interactions.c}
  \item{...}{additional arguments (not currently used)}
}
\value{
  \item{yhat}{Vector of predictions for each observation. For logistic model, these are the
estimated probabilities.}
}
\references{Bien, J., Taylor, J., Tibshirani, R., (2013) "A Lasso for Hierarchical Interactions." Annals of Statistics. 41(3). 1111-1141.}
\author{Jacob Bien and Robert Tibshirani}
\seealso{\link{hierNet}, \link{hierNet.path} }
\examples{
set.seed(12)
x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
newx=matrix(rnorm(100*10),ncol=10)
fit=hierNet(x,y,lam=50)
yhat=predict(fit,newx)

fit=hierNet.path(x,y)
yhat=predict(fit,newx)
}
