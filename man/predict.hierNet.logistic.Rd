\name{predict.hierNet.logistic}
\alias{predict.hierNet.logistic}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Prediction function for hierNet.logistic.}
\description{
A function to perform prediction, using an x matrix and the output of
the "hierNet.logistic" function or "hierNet.logistic.path".
}
\usage{
\method{predict}{hierNet.logistic}(object, newx, newzz=NULL,...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{The results of a call to the "hierNet.logistic" or "hierNet.logistic.path" or
 function. The
    coefficients that are part of this object will be used for
    making predictions.}
  \item{newx}{The new x at which predictions should be made. Can be a
    vector 
     or a matrix  (one observation per row).}
  \item{newzz}{Optional matrix of products of columns of newx, computed by compute.interactions.c}
 \item{...}{additional arguments (not currently used)}

}
\value{
  \item{yhat}{Matrix of predictions (probabilities), one row per observation}
}
\references{Bien, J., Taylor, J., Tibshirani, R., (2013) "A Lasso for Hierarchical Interactions." Annals of Statistics. 41(3). 1111-1141.}
\author{Jacob Bien and Robert Tibshirani}
\seealso{\link{hierNet.logistic}, \link{hierNet.logistic.path} }
\examples{
set.seed(12)
x=matrix(rnorm(100*10),ncol=10)
x=scale(x,TRUE,TRUE)
y=x[,1]+2*x[,2]+ x[,1]*x[,2]+3*rnorm(100)
y=1*(y>0)
newx=matrix(rnorm(100*10),ncol=10)
fit=hierNet.logistic(x,y,lam=5)
yhat=predict(fit,newx)

fit=hierNet.logistic.path(x,y)
yhat=predict(fit,newx)
}
