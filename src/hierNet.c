/*
Find optimal alpha for onerow optimization in ADMM4 within prox of
generalized gradient descent.
*/

#include <R.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))
#define abs(a) (((a) < 0) ? -(a) : (a))
#define ut(j, k, p) (p * j - j * (j + 1) / 2 + k - j - 1) // note: must have j < k
#define utd(j, k, p) (ut(j, (k+1), (p+1))) // note: must have j <= k (this indexes uppertri WITH diagonal)
#define SMALL 1e-80

void compute_yhat_zz(double *x, int n, int p, double *zz, int diagonal,
		     double *th, double *bp, double *bn, double *yhat);
void compute_phat_zz(double *x, int n, int p, double *zz, int diagonal, double b0, double *th, double *bp, double *bn, 
		     double *phat);
void ggdescent(double *x, int n, int p, double *zz, int diagonal, double *y, 
	       double lamL1, double lamL2, double rho, double *V, int maxiter, 
	       double *curth, double *curbp, double *curbn,
	       double t, int stepwindow, double backtrack, double tol, int trace,
	       double *th, double *bp, double *bn);
void ggdescent_logistic(double *x, int n, int p, double *zz, int diagonal, double *y, 
			double lamL1, double lamL2, double rho, double *V, int maxiter, 
			double *curb0, double *curth, double *curbp, double *curbn,
			double t, int stepwindow, double backtrack, double tol, int trace,
			double *b0, double *th, double *bp, double *bn);
void ggstep(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
	    double rho, double *V, double *curth, double *curbp, double *curbn,
	    double t, double backtrack, double *th, double *bp, double *bn, double *ttaken, double *maxabsdel);
void ggstep_logistic2(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
		     double rho, double *V, double *curb0, double *curth, double *curbp, double *curbn,
		     double t, double backtrack, double *b0, double *th, double *bp, double *bn, 
		     double *ttaken, double *maxabsdel);
void prox_zz(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
	     double rho, double *V, double *curth, double *curbp, double *curbn,
	     double t, double *th, double *bp, double *bn);
void prox_zz_logistic(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
		      double rho, double *V, double *curb0, double *curth, double *curbp, double *curbn,
		      double t, double *b0, double *th, double *bp, double *bn);
void prox_zz_given_r(double *x, int n, int p, double *zz, int diagonal, double *r, double lamL1, double lamL2,
		     double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn);
void prox_zz_given_r_nodiag(double *x, int n, int p, double *zz, double *r, double lamL1, double lamL2,
		     double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn);
void prox_zz_given_r_withdiag(double *x, int n, int p, double *zz, double *r, double lamL1, double lamL2,
		     double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn);
double prox_objective(double *x, int n, int p, double *zz, int diagonal, double *r, double lamL1, double lamL2,
		      double rho, double *V, double *curth, double *curbp, double *curbn,
		      double t, double *th, double *bp, double *bn);
void onerow(double *a, int q, double *b, double c, double mu,
	    double *th, double *beta, double *alpha);
void onerow_withdiag(double *a, int q, int jj, double *b, double c, double mu,
		     double *th, double *beta, double *alpha);
double f(double alpha, double *a, int q, double *b, double c, double mu);
double f_withdiag(double alpha, double *a, int q, int jj, double *b, double c, double mu);
void compute_dot_grad_del(double *zz, int diagonal, int n, int p, double *r, double *del, double *dotprod);

 // matrix functions
void PrintMatrix(double *, int, int);
void ComputeCrossProd(double *, int, int, double *, int, double *);

void onerow_R(double *a, int *q, double *b, double *c, double *mu,
	      double *th, double *beta, double *alpha) {
  onerow(a, *q, b, *c, *mu, th, beta, alpha);
}

void onerow_withdiag_R(double *a, int *q, int *jj, double *b, double *c, double *mu,
	    double *th, double *beta, double *alpha) {
  onerow_withdiag(a, *q, *jj, b, *c, *mu, th, beta, alpha);
}

void compute_yhat_zz_R(double *x, int *n, int *p, double *zz, int *diagonal,
		    double *th, double *bp, double *bn, double *yhat) {
  compute_yhat_zz(x, *n, *p, zz, *diagonal, th, bp, bn, yhat);
}

void compute_phat_zz_R(double *x, int *n, int *p, double *zz, int *diagonal,
		       double *b0, double *th, double *bp, double *bn, double *phat) {
  compute_phat_zz(x, *n, *p, zz, *diagonal, *b0, th, bp, bn, phat);
}

void prox_zz_R(double *x, int *n, int *p, double *zz, int *diagonal, double *y, double *lamL1, double *lamL2,
	    double *rho, double *V, double *curth, double *curbp, double *curbn,
	    double *t, double *th, double *bp, double *bn) {
  prox_zz(x, *n, *p, zz, *diagonal, y, *lamL1, *lamL2, *rho, V, curth, curbp, curbn, *t, th, bp, bn);
}

void ggstep_zz_R(double *x, int *n, int *p, double *zz, int *diagonal, double *y, double *lamL1, double *lamL2,
		 double *rho, double *V, double *curth, double *curbp, double *curbn,
		 double *t, double *backtrack, double *th, double *bp, double *bn, double *ttaken, double *maxabsdel) {
  ggstep(x, *n, *p, zz, *diagonal, y, *lamL1, *lamL2, *rho, V, curth, curbp, curbn, *t, *backtrack, th, bp, bn, ttaken, maxabsdel);
}

void ggdescent_R(double *x, int *n, int *p, double *zz, int *diagonal, double *y,
		 double *lamL1, double *lamL2, double *rho, double *V, int *maxiter,
		 double *curth, double *curbp, double *curbn,
		 double *t, int *stepwindow, double *backtrack, double *tol, int *trace,
		 double *th, double *bp, double *bn) {
  ggdescent(x, *n, *p, zz, *diagonal, y, *lamL1, *lamL2, *rho, V, *maxiter, curth, curbp, curbn, 
  	    *t, *stepwindow, *backtrack, *tol, *trace, th, bp, bn);
}

void ggdescent_logistic_R(double *x, int *n, int *p, double *zz, int * diagonal, double *y, 
			  double *lamL1, double *lamL2, double *rho, double *V, int *maxiter, 
			  double *curb0, double *curth, double *curbp, double *curbn,
			  double *t, int *stepwindow, double *backtrack, double *tol, int *trace,
			  double *b0, double *th, double *bp, double *bn) {
  ggdescent_logistic(x, *n, *p, zz, *diagonal, y, *lamL1, *lamL2, *rho, V, *maxiter, curb0, curth, curbp, curbn,
		     *t, *stepwindow, *backtrack, *tol, *trace, b0, th, bp, bn);
}

void compute_dot_grad_del(double *zz, int diagonal, int n, int p, double *r, double *del, double *dotprod) {
  /* Given a p by p matrix del, computes <del, grad_th>=sum_{jk}del_{jk}grad_th_{jk} 
   where grad_th_{jk} = -(1/2)sum_{i} z_{i,jk}r_i.  Note that since del_{jk} may often
   be very sparse, we don't need calculate very much of grad_th.
  */
  int i, j, k, ii;
  double dd, grad, t;
  *dotprod = 0.0;
  if (diagonal) {
    for (j = 0; j < p-1; j++) {
      for (k = j+1; k < p; k++) {
	dd = del[j + p * k] + del[k + p * j];
	if (dd != 0) {
	  // compute grad_th_{jk}:
	  grad = 0;
	  ii = utd(j, k, p);
	  for (i = 0; i < n; i++) {
	    t = -zz[i + n*ii] * r[i] / 2;
	    grad += t;
	  }
	  *dotprod += dd * grad;
	}
      }
    }
    for (j = 0; j < p; j++) {
      dd = del[j + p * j];
      if (dd != 0) {
	// compute grad_th_{jj}:
	grad = 0;
	ii = utd(j, j, p);
	for (i = 0; i < n; i++) {
	  t = -zz[i + n*ii] * r[i]; // I think no 1/2 is correct here.
	  grad += t;
	}
	*dotprod += dd * grad;
      }
    }
  } else {
    // no diagonal case
    for (j = 0; j < p-1; j++) {
      for (k = j+1; k < p; k++) {
	dd = del[j + p * k] + del[k + p * j];
	if (dd != 0) {
	  // compute grad_th_{jk}:
	  grad = 0;
	  ii = ut(j, k, p);
	  for (i = 0; i < n; i++) {
	    t = -zz[i + n*ii] * r[i] / 2;
	    grad += t;
	  }
	  *dotprod += dd * grad;
	}
      }
    }
  }
}

void compute_yhat_zz(double *x, int n, int p, double *zz, int diagonal, double *th, double *bp, double *bn, 
		  double *yhat) {
  /* 
     if diagonal:
     x: n by p
     zz: n by cp2 + p
     th: p^2
     bp, bn: p

     else:
     x: n by p
     zz: n by cp2
     th: p^2 (on diagonals assumed to be 0)
     bp, bn: p
   */
  int i, j, k, jj;
  double b;
  for (i = 0; i < n; i++)
    yhat[i] = 0;
  for (j = 0; j < p; j++) {
    b = bp[j] - bn[j];
    if (b != 0) {
      for (i = 0; i < n; i++) {
	yhat[i] += x[i + n * j] * b;
      }
    }
  }
  if (diagonal) {
    // include diagonal case
    for (j = 0; j < p-1; j++) {
      for (k = j + 1; k < p; k++) {
	b = th[j + p * k] + th[k + p * j];
	if (b != 0) {
	  jj = utd(j, k, p);
	  for (i = 0; i < n; i++) {
	    yhat[i] += zz[i + n * jj] * b / 2;
	  }
	}
      }
    }
    for (j = 0; j < p; j++) {
      b = th[j + p * j];
      jj = utd(j, j, p);
      if (b != 0) {
	for (i = 0; i < n; i++){
	  yhat[i] += zz[i + n * jj] * b;
	}
      }
    }
  } else {
    // no diagonal case
    for (j = 0; j < p-1; j++) {
      for (k = j + 1; k < p; k++) {
	b = th[j + p * k] + th[k + p * j];
	if (b != 0) {
	  jj = ut(j, k, p);
	  for (i = 0; i < n; i++) {
	    yhat[i] += zz[i + n * jj] * b / 2;
	  }
	}
      }
    }
  }
}

void compute_phat_zz(double *x, int n, int p, double *zz, int diagonal, double b0, double *th, double *bp, double *bn, 
		  double *phat) {
  int i;
  compute_yhat_zz(x, n, p, zz, diagonal, th, bp, bn, phat);
  for (i = 0; i < n; i++)
    phat[i] = 1/(1 + exp(-b0-phat[i]));
}

void ComputeInteractionsWithIndices(double *x, int *p_n, int *p_p, double *zz, 
					   int *i1, int *i2) {
  /* Computes uncentered zz
   Note: does not allocate memory for zz.  This must be done prior to calling function.
   Let zz denote the n by cp2 matrix (whereas z denotes n by p^2 matrix).
  */
  int n = *p_n;
  int p = *p_p;
  int i, j, k, jj;
  int counter = 0;
  for (j = 0; j < p - 1; j++) {
    for (k = j + 1; k < p; k++) {
      jj = ut(j, k, p);
      for (i = 0; i < n; i++) {
	zz[i + n * jj] = x[i + n * j] * x[i + n * k];
      }
      i1[counter] = j + 1;
      i2[counter++] = k + 1;
    }
  }
}

void ComputeInteractionsWithDiagWithIndices(double *x, int *p_n, int *p_p, double *zz, 
					   int *i1, int *i2) {
  /* Computes uncentered zz
   Note: does not allocate memory for zz.  This must be done prior to calling function.
   Let zz denote the n by cp2+p matrix (whereas z denotes n by p^2 matrix).
  */
  int n = *p_n;
  int p = *p_p;
  int i, j, k, jj;
  int counter = 0;
  for (j = 0; j < p; j++) {
    for (k = j; k < p; k++) {
      jj = utd(j, k, p);
      for (i = 0; i < n; i++) {
	zz[i + n * jj] = x[i + n * j] * x[i + n * k];
      }
      i1[counter] = j + 1;
      i2[counter++] = k + 1;
    }
  }
}


void ComputeFullInteractions(double *x, int *p_n, int *p_p, double *z) {
  /* Computes n by p^2 matrix of uncentered interactions.  Somewhat wasteful, but sometimes easier
     to work with.
   Note: does not allocate memory for zz.  This must be done prior to calling function.
  */
  int n = *p_n;
  int p = *p_p;
  int i, j, k;
  for (j = 0; j < p - 1; j++) {
    for (k = j + 1; k < p; k++) {
      for (i = 0; i < n; i++) {
	z[i + n * (j + p * k)] = x[i + n * j] * x[i + n * k];
	z[i + n * (k + p * j)] = z[i + n * (j + p * k)];
      }
    }
  }
  for (j = 0; j < p; j++)
    for (i = 0; i < n; i++)
      z[i + n * j * (p + 1)] = x[i + n * j] * x[i + n * j];
}


void ggdescent(double *x, int n, int p, double *zz, int diagonal, double *y, 
	       double lamL1, double lamL2, double rho, double *V, int maxiter, 
	       double *curth, double *curbp, double *curbn,
	       double t, int stepwindow, double backtrack, double tol, int trace,
	       double *th, double *bp, double *bn) {
  /*
     This function performs generalized gradient descent as described in the ADMM4 pdf.

     Initial point is (curth,curbp,curbn).  Final point is (th,bp,bn).

     t is the initial step size.  Every iteration (after 'stepwindow' iterations)
     it reduces the step size to the largest step size chosen by backtracking line search
     in the last 'stepwindow' iterations.
  */
  int l, ll, nswap = 0;
  double tt, ttaken; // proposed step size and step size chosen
  double ttt[stepwindow]; /* last 'stepwindow' stepsizes chosen by BLS. */
  double maxabsdel;
  double *temp;
  for (l = 0; l < maxiter; l++) {
    if (trace > 0) Rprintf("%d", l+1);
    if (l < stepwindow)
      tt = t;
    else {
      // find max of last stepwindow stepsizes:
      tt = 0;
      for (ll = 0; ll < stepwindow; ll++)
	if (ttt[ll] > tt)
	  tt = ttt[ll];
    }
    ggstep(x, n, p, zz, diagonal, y, lamL1, lamL2, rho, V, curth, curbp, curbn, tt, backtrack, th, bp, bn, &ttaken, &maxabsdel);
    if (maxabsdel < tol) {
      // note: don't need to update (curth,curbp,curbn)
      Rprintf("GG converged in %d iterations.\n", l+1);
      break;
    }
    if (l < maxiter - 1) {
      /* 
	 We have just taken a step and need to update (curth,curbp,curbn) since we have yet to converge.
	 Rather than copy over all the elements from (th,bp,bn) to (curth,curbp,curbn),
	 simply have the latter point to the former.  But we need space allocated for (th,bp,bn) for future
	 iterations... therefore we use what used to be pointed to by (curth,curbp,curbn)!

	 Note that if l == maxiter - 1, then we want to leave (th,bp,bn) as is and don't care about updating cur.
      */
      temp = curth; curth = th; th = temp;
      temp = curbp; curbp = bp; bp = temp;
      temp = curbn; curbn = bn; bn = temp;
      nswap++;
    }
    ttt[l % stepwindow] = ttaken;
  }

  if (nswap % 2 == 1) {
    temp = curth; curth = th; th = temp;
    temp = curbp; curbp = bp; bp = temp;
    temp = curbn; curbn = bn; bn = temp;
    int j, jj;
    for (j = 0; j < p; j++) {
      bp[j] = curbp[j];
      bn[j] = curbn[j];
    }
    for (jj = 0; jj < p * p; jj++)
      th[jj] = curth[jj];
  }
}

void ggdescent_logistic(double *x, int n, int p, double *zz, int diagonal, double *y, 
			double lamL1, double lamL2, double rho, double *V, int maxiter, 
			double *curb0, double *curth, double *curbp, double *curbn,
			double t, int stepwindow, double backtrack, double tol, int trace,
			double *b0, double *th, double *bp, double *bn) {
  /*
     This function performs generalized gradient descent as described in the ADMM4 pdf.  However,
     as described in logistic.pdf, it differs in (a) the way the residual is calculated and 
     (b) the handling of the intercept.

     Initial point is (curb0,curth,curbp,curbn).  Final point is (b0,th,bp,bn).

     t is the initial step size.  Every iteration (after 'stepwindow' iterations)
     it reduces the step size to the largest step size chosen by backtracking line search
     in the last 'stepwindow' iterations.
  */
  int l, ll, nswap = 0;
  double tt, ttaken; // proposed step size and step size chosen
  double ttt[stepwindow]; /* last 'stepwindow' stepsizes chosen by BLS. */
  double maxabsdel;
  double *temp;
  for (l = 0; l < maxiter; l++) {
    if (trace > 0) Rprintf("%d", l+1);
    if (l < stepwindow)
      tt = t;
    else {
      // find max of last stepwindow stepsizes:
      tt = 0;
      for (ll = 0; ll < stepwindow; ll++)
	if (ttt[ll] > tt)
	  tt = ttt[ll];
    }
    ggstep_logistic2(x, n, p, zz, diagonal, y, lamL1, lamL2, rho, V, curb0, curth, curbp, curbn, 
		     tt, backtrack, b0, th, bp, bn, &ttaken, &maxabsdel);
    if (maxabsdel < tol) {
      // note: don't need to update (curth,curbp,curbn)
      Rprintf("GG converged in %d iterations.\n", l+1);
      break;
    }
    if (l < maxiter - 1) {
      //	 We have just taken a step and need to update (curth,curbp,curbn) since we have yet to converge.
      // Rather than copy over all the elements from (th,bp,bn) to (curth,curbp,curbn),
      // simply have the latter point to the former.  But we need space allocated for (th,bp,bn) for future
      // iterations... therefore we use what used to be pointed to by (curth,curbp,curbn)!

	// Note that if l == maxiter - 1, then we want to leave (th,bp,bn) as is and don't care about updating cur.
      
      temp = curth; curth = th; th = temp;
      temp = curbp; curbp = bp; bp = temp;
      temp = curbn; curbn = bn; bn = temp;
      temp = curb0; curb0 = b0; b0 = temp;
      nswap++;
    }
    ttt[l % stepwindow] = ttaken;
  }

  if (nswap % 2 == 1) {
    temp = curth; curth = th; th = temp;
    temp = curbp; curbp = bp; bp = temp;
    temp = curbn; curbn = bn; bn = temp;
    temp = curb0; curb0 = b0; b0 = temp;
    int j, jj;
    for (j = 0; j < p; j++) {
      bp[j] = curbp[j];
      bn[j] = curbn[j];
    }
    for (jj = 0; jj < p * p; jj++)
      th[jj] = curth[jj];
    *b0 = *curb0;
  }
}

void ggstep(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
	    double rho, double *V, double *curth, double *curbp, double *curbn,
	    double t, double backtrack, double *th, double *bp, double *bn, double *ttaken, double *maxabsdel) {
  /*
    t: initial step size
    ttaken: step size taken (chosen via backtracking line search)
    maxabsdel: the most any parameter changed.
  */
  if (backtrack >= 1 || backtrack <= 0)
    Rprintf("WARNING!  'backtrack' must be in (0,1) for backtracking to work!");
  int i, j, jj;
  double tt, left, right0, right, dotprod, temp;
  double r[n], curr[n], cm[p], delbp[p], delbn[p], delth[p*p];
  int ismajorizing = 0;
  // compute current residual, curr.  Also, form the constant part of 'right':
  compute_yhat_zz(x, n, p, zz, diagonal, curth, curbp, curbn, curr);
  right0 = 0;
  for (i = 0; i < n; i++) {
    curr[i] = y[i] - curr[i];
    right0 += curr[i] * curr[i];
  }
  right0 /= 2;
  // compute cm = X^Tr:
  ComputeCrossProd(x, n, p, curr, 1, cm);
  tt = t;
  // now start backtracking steps...
  while (tt > SMALL) {
    // try taking a step of size tt:
    prox_zz(x, n, p, zz, diagonal, y, lamL1, lamL2, rho, V, curth, curbp, curbn, tt, th, bp, bn);
    // compute this step's residual, r.  Also, compute 'left':
    left = 0;
    compute_yhat_zz(x, n, p, zz, diagonal, th, bp, bn, r);
    for (i = 0; i < n; i++) {
      r[i] = y[i] - r[i];
      left += r[i] * r[i];
    }
    left /= 2;
    // now compute right:
    right = right0;
    // compute change in parameters (from cur):
    for (j = 0; j < p; j++) {
      delbp[j] = bp[j] - curbp[j];
      delbn[j] = bn[j] - curbn[j];
    }
    for (jj = 0; jj < p*p; jj++)
      delth[jj] = th[jj] - curth[jj];
    compute_dot_grad_del(zz, diagonal, n, p, curr, delth, &dotprod);
    right += dotprod;
    temp = 0;
    for (j = 0; j < p; j++) {
      right += cm[j] * (delbn[j] - delbp[j]);
      temp += delbp[j] * delbp[j] + delbn[j] * delbn[j];
    }
    for (jj = 0; jj < p*p; jj++)
      temp += delth[jj] * delth[jj];
    right += temp / (2 * tt);
    if (left <= right) {
      ismajorizing = 1;
      break;
    }
    tt *= backtrack;
  }
  if (ismajorizing) {
    *ttaken = tt;
    // compute the biggest absolute change in any parameter...
    *maxabsdel = 0;
    for (j = 0; j < p; j++) {
      delbp[j] = abs(delbp[j]);
      delbn[j] = abs(delbn[j]);
      if (*maxabsdel < delbp[j]) *maxabsdel = delbp[j];
      if (*maxabsdel < delbn[j]) *maxabsdel = delbn[j];
    }
    for (jj = 0; jj < p*p; jj++) {
      delth[jj] = abs(delth[jj]);
      if (*maxabsdel < delth[jj]) *maxabsdel = delth[jj];
    }
  } else {
    Rprintf("WARNING! Stepsize is below %f, and yet GG's spherical approx is not majorizing!\n", SMALL);
  }
}

void prox_zz(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
	     double rho, double *V, double *curth, double *curbp, double *curbn,
	     double t, double *th, double *bp, double *bn) {
  /* 
     zz refers to the cross products matrix with only cp2 columns rather than p^2.  
     
     curth, curbp, curbn are not modified.  Rather th, bp, bn are. 
  */
  int i;
  double r[n];
  compute_yhat_zz(x, n, p, zz, diagonal, curth, curbp, curbn, r);
  for (i = 0; i < n; i++)
    r[i] = y[i] - r[i];
  prox_zz_given_r(x, n, p, zz, diagonal, r, lamL1, lamL2, rho, V, curth, curbp, curbn, t, th, bp, bn);
}

double compute_logistic_loss(double *x, int n, int p, double *zz, int diagonal, double *y, double b0, double *th, double *bp, double *bn) {
  /* 
     Compute negative log-likelihood: -sum_{i=1}^n y_i log p_i + (1 - y_i) log(1 - p_i) = log(1+e^{-z_i A_i}
     where p_i = 1/(1+e^{-A_i}) and z_i = 2y_i - 1
  */
  int i;
  double loss = 0;
  double yhat[n];
  compute_yhat_zz(x, n, p, zz, diagonal, th, bp, bn, yhat);
  for (i = 0; i < n; i++) {
    loss += log(1 + exp(-(2 * y[i] - 1) * (b0 + yhat[i])));
  }
  return loss;
}

double penalty(double *x, int n, int p, int diagonal, double lamL1, double lamL2, double rho, double *V,
	       double b0, double *th, double *bp, double *bn) {
  /* includes ADMM terms as well as l1 and l2 penalties. */
  int j, jj;
  double pen = 0, pen1 = 0, pen2 = 0, pen3 = 0;
  for (jj = 0; jj < p*p; jj++) {
    pen += abs(th[jj]);
    pen1 += th[jj] * th[jj];
    pen2 += th[jj] * V[jj];
  }
  if (diagonal) {
    for (j = 0; j < p; j++)
      pen += abs(th[j+p*j]);
  }
  pen /= 2;
  for (j = 0; j < p; j++)
    pen += bp[j] + bn[j];
  pen *= lamL1;
  pen += rho * pen1 / 2;
  pen += pen2;
  if (lamL2 != 0) {
    // elastic net term
    for (j = 0; j < p; j++)
      pen3 += bp[j] * bp[j] + bn[j] * bn[j];
    pen += lamL2 * (pen3 + pen1);
  }
  return pen; 
}

double compute_logistic_objective(double *x, int n, int p, double *zz, int diagonal, double *y,
				  double lamL1, double lamL2, double rho, double *V,
				  double b0, double *th, double *bp, double *bn) {
  /* 
     negative log-likelihood + penalty terms
  */
  return compute_logistic_loss(x, n, p, zz, diagonal, y, b0, th, bp, bn) + penalty(x, n, p, diagonal, lamL1,lamL2, rho, V, b0, th, bp, bn);
}


void ggstep_logistic2(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
		     double rho, double *V, double *curb0, double *curth, double *curbp, double *curbn,
		     double t, double backtrack, double *b0, double *th, double *bp, double *bn, 
		     double *ttaken, double *maxabsdel) {
  /*
    t: initial step size
    ttaken: step size taken (chosen via backtracking line search)
    maxabsdel: the most any parameter changed.
  */
  if (backtrack >= 1 || backtrack <= 0)
    Rprintf("WARNING!  'backtrack' must be in (0,1) for backtracking to work!");
  int i, j, jj;
  double tt, left, curloss, loss, right, sqnorm, temp, sumcurr;
  double curr[n], delr[n], delyhat[n], delb0, delbp[p], delbn[p], delth[p*p];
  int ismajorizing = 0;
  // compute current residual, curr.  Also, form the constant part of 'right':
  compute_phat_zz(x, n, p, zz, diagonal, *curb0, curth, curbp, curbn, curr);
  sumcurr = 0;
  for (i = 0; i < n; i++) {
    curr[i] = y[i] - curr[i];
    sumcurr += curr[i];
  }
  curloss = compute_logistic_loss(x, n, p, zz, diagonal, y, *curb0, curth, curbp, curbn);// see logistic.pdf
  tt = t;
  // now start backtracking steps... until logistic loss <= spherical approx when eval'ed at new point
  while (tt > SMALL) {
    // try taking a step of size tt:
    
    //Rprintf("Q(*|x^k-1)=%f\n",compute_logistic_gg_majorizer_obj(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, tt, *curb0, curth, curbp, curbn));
    *b0 = *curb0 + tt * sumcurr;
    //Rprintf("Q(*|x^k-1)=%f\n",compute_logistic_gg_majorizer_obj(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, tt, *b0, curth, curbp, curbn));
    prox_zz_logistic(x, n, p, zz, diagonal, y, lamL1, lamL2, rho, V, curb0, curth, curbp, curbn, tt, b0, th, bp, bn);
    //Rprintf("Q(*|x^k-1)=%f\n",compute_logistic_gg_majorizer_obj(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, tt, *b0, th, bp, bn));
    loss = compute_logistic_loss(x, n, p, zz, diagonal, y, *b0, th, bp, bn);
    // compute ||coefs^k-coefs^{k-1}||^2/(2t) and  yhat^{k} - yhat^{k-1} (see logistic.pdf):
    delb0 = *b0 - *curb0;
    sqnorm = delb0 * delb0;
    for (j = 0; j < p; j++) {
      delbp[j] = bp[j] - curbp[j];
      delbn[j] = bn[j] - curbn[j];
      sqnorm += delbp[j] * delbp[j] + delbn[j] * delbn[j];
    }
    for (jj = 0; jj < p*p; jj++) {
      delth[jj] = th[jj] - curth[jj];
      sqnorm += delth[jj] * delth[jj];
    }
    sqnorm /= 2 * tt;
    compute_yhat_zz(x, n, p, zz, diagonal, delth, delbp, delbn, delyhat);// dely = y^{k} - y^{k-1}
    for (i = 0; i < n; i++)
      delyhat[i] += delb0;
    if (abs(curloss - loss) >= 0.0001 * loss) { // 0.0001 is an arbitrary choice!
      // this is the numerically stable case (see TFOCS paper, page 25)
      // now compute right: (see logistic.pdf for details)
      left = loss;
      right = curloss + sqnorm;
      for (i = 0; i < n; i++)
	right -= delyhat[i] * curr[i];
      //Rprintf("This (left:%f) (right:%f)\n", left, right);
    }
    else {
      //Rprintf("That!!! %f approx %f\n", curloss, loss);
      // this is the case in which usual backtracking criterion becomes numerically unstable (see logistic.pdf)
      compute_phat_zz(x, n, p, zz, diagonal, *b0, th, bp, bn, delr);// this is p^k
      for (i = 0; i < n; i++)
	delr[i] = y[i] - delr[i] - curr[i];// delr = (y - p^k) - r^{k-1} = r^k - r^{k-1}
      left = 0;
      for (i = 0; i < n; i++)
	left += delyhat[i] * delr[i];
      left = abs(left);
      right = sqnorm;
    }
    if (left <= right) {
      //Rprintf(" tt=%f\n",tt);
      /*
      Rprintf("f(x^k-1)=%f\n",compute_logistic_objective(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn));
      Rprintf("Q(x^k-1|x^k-1)=%f\n",compute_logistic_gg_majorizer_obj(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, tt, *curb0, curth, curbp, curbn));
      Rprintf("Q(x^k|x^k-1)=%f\n",compute_logistic_gg_majorizer_obj(x, n, p, zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, tt, *b0, th, bp, bn));
      Rprintf("right + pen = %f\n", right + penalty(x, n, p, diagonal, lam, rho, V, *b0, th, bp, bn));      
      Rprintf("f(x^k)=%f\n", compute_logistic_objective(x, n, p, zz, diagonal, y, lam, rho, V, *b0, th, bp, bn));//compute_logistic_loss(x, n, p, zz, y, *b0, th, bp, bn));
      Rprintf("left + pen = %f\n", left + penalty(x, n, p, diagonal, lam, rho, V, *b0, th, bp, bn));
      */      
      ismajorizing = 1;
      break;
    }
    tt *= backtrack;
  }
  if (ismajorizing) {
    *ttaken = tt;
    // compute the biggest absolute change in any parameter...
    *maxabsdel = abs(*b0 - *curb0);
    for (j = 0; j < p; j++) {
      temp = abs(bp[j] - curbp[j]);
      if (*maxabsdel < temp) *maxabsdel = temp;
      temp = abs(bn[j] - curbn[j]);
      if (*maxabsdel < temp) *maxabsdel = temp;
    }
    for (jj = 0; jj < p*p; jj++) {
      temp = abs(th[jj] - curth[jj]);
      if (*maxabsdel < temp) *maxabsdel = temp;
    }
  } else {
    Rprintf("WARNING! Stepsize is below %f, and yet GG's spherical approx is not majorizing!\n", SMALL);
    Rprintf("...This is likely due to numerical instability in backtracking.\n");
  }
}


void prox_zz_logistic(double *x, int n, int p, double *zz, int diagonal, double *y, double lamL1, double lamL2,
		      double rho, double *V, double *curb0, double *curth, double *curbp, double *curbn,
		      double t, double *b0, double *th, double *bp, double *bn) {
  /* 
     zz refers to the cross products matrix with only cp2 columns rather than p^2.  
     
     curb0, curth, curbp, curbn are not modified.  Rather b0, th, bp, bn are. 

     see logistic.pdf for derivation of generalized gradient descent for logistic regression,
     justifying that the only change is that we compute r = y - phat rather than r = y - Xbetahat
  */
  int i;
  double r[n];
  compute_phat_zz(x, n, p, zz, diagonal, *curb0, curth, curbp, curbn, r);
  for (i = 0; i < n; i++)
    r[i] = y[i] - r[i];
  prox_zz_given_r(x, n, p, zz, diagonal, r, lamL1, lamL2, rho, V, curth, curbp, curbn, t, th, bp, bn);


  /*
  double proxbef = prox_objective(x, n, p, zz, diagonal, r, lam, rho, V, curth, curbp, curbn, t, curth, curbp, curbn);
  double proxaft = prox_objective(x, n, p, zz, diagonal, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn);
  //  double bef = compute_logistic_gg_majorizer_obj(x,n,p,zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, t, *b0, curth, curbp, curbn);
  //double aft = compute_logistic_gg_majorizer_obj(x,n,p,zz, diagonal, y, lam, rho, V, *curb0, curth, curbp, curbn, t, *b0, th, bp, bn);
  if (proxaft > proxbef) {
    //Rprintf("Prox problem! before:%f  after:%f\n",bef,aft);
    Rprintf("Prox problem!  before:%f, after:%f\n",proxbef,proxaft);
  }
  */
}

double prox_objective(double *x, int n, int p, double *zz, int diagonal, double *r, double lamL1, double lamL2,
		      double rho, double *V, double *curth, double *curbp, double *curbn,
		      double t, double *th, double *bp, double *bn) {
  int j, k, jj, kk;
  int cp2 = p * (p-1) / 2;
  if (diagonal) cp2 += p;
  double cm[p];
  double ci[cp2];
  double obj;
  ComputeCrossProd(x, n, p, r, 1, cm);
  ComputeCrossProd(zz, n, cp2, r, 1, ci);
  obj = 0;
  for (j = 0; j < p; j++) {
    obj += pow(bp[j] - (curbp[j]+t*cm[j]),2);
    obj += pow(bn[j] - (curbn[j]-t*cm[j]),2);
  }
  if (diagonal) {
    for (j = 0; j < p; j++) {
      for (k = 0; k < p; k++) {
	if (j == k) 
	  continue;
	jj = j + p * k;
	if (j < k)
	  kk = utd(j,k,p);
	else
	  kk = utd(k,j,p);
	obj += pow(th[jj] - curth[jj] - t * ci[kk]/2, 2);      
      }
    }
    for (j = 0; j < p; j++) {
      jj = j + p * j;
      obj += pow(th[jj] - curth[jj] - t * ci[utd(j,j,p)], 2);
    }
  } else {
    for (j = 0; j < p; j++) {
      for (k = 0; k < p; k++) {
	if (j == k) 
	  continue;
	jj = j + p * k;
	if (j < k)
	  kk = ut(j,k,p);
	else
	  kk = ut(k,j,p);
	obj += pow(th[jj] - curth[jj] - t * ci[kk]/2, 2);
      }
    }
  }
  //printf("zz[0]=%f ci[0]=%f cm[0]=%f rss=%f\n",zz[0],ci[0],cm[0],obj);
  obj /= 2 * t;
  obj += penalty(x, n, p, diagonal, lamL1, lamL2, rho, V, 0, th, bp, bn);
  return obj;
}

void prox_zz_given_r(double *x, int n, int p, double *zz, int diagonal, double *r, double lamL1, 
		     double lamL2, double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn) {
  if (diagonal)
    prox_zz_given_r_withdiag(x, n, p, zz, r, lamL1, lamL2, rho, V, curth, curbp, curbn, t, th, bp, bn);
  else
    prox_zz_given_r_nodiag(x, n, p, zz, r, lamL1, lamL2, rho, V, curth, curbp, curbn, t, th, bp, bn);
}

void prox_zz_given_r_nodiag(double *x, int n, int p, double *zz, double *r, double lamL1, double lamL2,
		     double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn) {
  /* 
     zz refers to the cross products matrix with only cp2 columns rather than p^2.  

     this solves the prox without computing r.  This makes it applicable for both
     linear regression and logistic regression losses.  It takes r instead of y.

     min_{bp,bn,th} ||bp - (curbp+t*X^Tr)||^2 + ||bn - (curbn-t*X^Tr)||^2
                    + ||th - (curth+t*zz^Tr)||^2 
                    + (2t)*lamL1*sum(bp+bn) + (2t)*V^Tth + (2t)*(rho/2)||th||^2 
		    + (2t)*(lamL1/2)||th||_1 + (2t)*lamL2*(sum(bp^2)+sum(bn^2)+sum(th^2))
     subject to bp>=0, bn>=0, ||th_j||_1 <= bp_j + bn_j

     See admm4.pdf for details.

     curth, curbp, curbn are not modified.  Rather th, bp, bn are. 
  */
  int j, k, jj, ii, kk;
  int cp2 = p * (p-1) / 2;
  double cm[p], a[p-1], b[2], row[p-1], beta[2], s[p-1];
  double ci[cp2];
  double c, mu, alpha, d;
  //double before_proxj;
  ComputeCrossProd(x, n, p, r, 1, cm);
  ComputeCrossProd(zz, n, cp2, r, 1, ci);
  d = 1 + 2 * t * lamL2;
  c = 1 + t * rho / d;
  mu = t * lamL1 / (2*d);
  for (j = 0; j < p; j++) { // this is a loop over rows of Theta
    ii = 0;
    // calculate a and b for this j:
    for (k = 0; k < p; k++) {
      jj = j + p * k;
      if (k == j) {
	ii = 1; // shift indexing back by one
	continue;
      }
      if (j < k)
	kk = ut(j,k,p);
      else
	kk = ut(k,j,p);
      a[k-ii] = (curth[jj] + t * (ci[kk]/2 - V[jj])) / (c*d);
      // take absolute value of a while retaining signs for later.
      if (a[k-ii] < 0) {
	s[k-ii] = -1;
	a[k-ii] *= -1;
      } else {
	s[k-ii] = 1;
      }
    }

    b[0] = (curbp[j] + t * (cm[j] - lamL1)) / d;
    b[1] = (curbn[j] - t * (cm[j] + lamL1)) / d;
    // do "onerow" for this (a, b) pair
    onerow(a, p-1, b, c, mu, row, beta, &alpha);
    /* This was for debugging:
      before_proxj = prox_objective(x, n, p, zz, 0, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn);
    */
    bp[j] = beta[0];
    bn[j] = beta[1];
    ii = 0;
    for (k = 0; k < p; k++) {
      if (k == j) {
	ii = 1;
	continue;
      }
      th[j + p * k] = row[k-ii] * s[k-ii];
    }
    /*
    if (prox_objective(x, n, p, zz, 0, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn) > before_proxj) {
      Rprintf("Prox objective rose for (j = %d) from %f to %f!!!\n",j,before_proxj,prox_objective(x, n, p, zz, 0, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn));
    }
    */
  }
}

void prox_zz_given_r_withdiag_R(double *x, int *n, int *p, double *zz, double *r, double *lamL1, double *lamL2,
		     double *rho, double *V, double *curth, double *curbp, double *curbn,
		     double *t, double *th, double *bp, double *bn) {
  prox_zz_given_r_withdiag(x, *n, *p, zz, r, *lamL1, *lamL2, *rho, V, curth, curbp, curbn, *t, th, bp, bn);
}

void prox_zz_given_r_withdiag(double *x, int n, int p, double *zz, double *r, double lamL1, double lamL2,
		     double rho, double *V, double *curth, double *curbp, double *curbn,
		     double t, double *th, double *bp, double *bn) {
  /* 
     zz refers to the cross products matrix with only cp2 columns rather than p^2.  

     this solves the prox without computing r.  This makes it applicable for both
     linear regression and logistic regression losses.  It takes r instead of y.

     min_{bp,bn,th} ||bp - (curbp+t*X^Tr)||^2 + ||bn - (curbn-t*X^Tr)||^2
                    + ||th - (curth+t*zz^Tr)||^2 
                    + (2t)*lamL1*sum(bp+bn) + (2t)*V^Tth + (2t)*(rho/2)||th||^2 
		    + (2t)*(lamL1/2)||th||_1 + (2t)*(lamL1/2)||diag(th)||_1 
		    + (2t)*lamL2*(sum(bp^2)+sum(bn^2)+sum(th^2))
     subject to bp>=0, bn>=0, ||th_j||_1 <= bp_j + bn_j

     See admm4.pdf for details.

     curth, curbp, curbn are not modified.  Rather th, bp, bn are. 
  */
  int j, k, jj, kk;
  int cp2 = p * (p+1) / 2;
  double cm[p], a[p], b[2], row[p], beta[2], s[p];
  double ci[cp2];
  double c, mu, alpha, d;
  //double before_proxj;
  ComputeCrossProd(x, n, p, r, 1, cm);
  ComputeCrossProd(zz, n, cp2, r, 1, ci);

  d = 1 + 2 * t * lamL2;
  c = 1 + t * rho / d;
  mu = t * lamL1 / (2*d);
  for (j = 0; j < p; j++) { // this is a loop over rows of Theta
    // calculate a and b for this j:
    //Rprintf("j=%d\n",j);
    //printf("a:\n");
    for (k = 0; k < p; k++) {
      jj = j + p * k;
      if (j == k) {
	kk = utd(j,j,p);
	a[k] = (curth[jj] + t * (ci[kk] - V[jj])) / (c*d);
	//printf("curth[jj]=%f  V[jj]=%f",curth[jj],V[jj]);
	//printf("ci[kk]=%fn",ci[kk]);
      } else {
	if (j < k)
	  kk = utd(j,k,p);
	else if (j > k)
	  kk = utd(k,j,p);
	a[k] = (curth[jj] + t * (ci[kk]/2 - V[jj])) / (c*d);
	//printf("curth[jj]=%f  V[jj]=%f",curth[jj],V[jj]);
	//printf("ci[kk]=%f\n",ci[kk]);
      }
      // take absolute value of a while retaining signs for later.
      if (a[k] < 0) {
	s[k] = -1;
	a[k] *= -1;
      } else {
	s[k] = 1;
      }
    }
    
    b[0] = (curbp[j] + t * (cm[j] - lamL1)) / d;
    b[1] = (curbn[j] - t * (cm[j] + lamL1)) / d;
    // do "onerow" for this (a, b) pair
    onerow_withdiag(a, p, j, b, c, mu, row, beta, &alpha);
    // This is for debugging:
    //before_proxj = prox_objective(x, n, p, zz, 1, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn);
    bp[j] = beta[0];
    bn[j] = beta[1];
    for (k = 0; k < p; k++) {
      th[j + p * k] = row[k] * s[k];
    }
    // This is for debugging:
    /*
    if (prox_objective(x, n, p, zz, 1, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn) > before_proxj) {
      printf("\nalpha=%f",alpha);
      Rprintf("Prox objective rose for (j = %d) from %f to %f!!!\n",j,before_proxj,prox_objective(x, n, p, zz, 1, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn));
    }
    */
    //printf("obj=%f\n",prox_objective(x, n, p, zz, 1, r, lam, rho, V, curth, curbp, curbn, t, th, bp, bn));
  }
}

void onerow(double *a, int q, double *b, double c, double mu,
	    double *th, double *beta, double *alpha) {
  // assume a_j >= 0
  int j, I;
  double b1, b2, lower; // b1 <= b2
  double val, ffold, ff, prev, cur;
  for (j = 0; j < q; j++) {
    th[j] = c * a[j] - mu;
  }
  R_rsort(th, q);
  // check if th is entirely 0:
  if (th[q-1] <= 0) { // max(a) <= c / mu
    // entire row of theta is zero, without help from hierarchy
    for (j = 0; j < q; j++)
      th[j] = 0;
    *alpha = 0;
    beta[0] = max(b[0], 0);
    beta[1] = max(b[1], 0);
    return;
  }
  if (b[0] >= b[1]) {
    b1 = b[1];
    b2 = b[0];
  } else {
    b1 = b[0];
    b2 = b[1];
  }
  if (th[q-1] <= -b2) { // max(a) <= (c - b2) / mu
    // entire row of theta is zero thanks to hierarchy
    for (j = 0; j < q; j++)
      th[j] = 0;
    *alpha = th[q-1];
    beta[0] = 0;
    beta[1] = 0;
    return;
  }
  // end screening.
  ffold = f(0, a, q, b, c, mu);
  //printf("f(0)=%f\n", ffold);
  if (ffold <= 0) {
    // Unconstrained solution is feasible!
    val = mu / c;
    for (j = 0; j < q; j++)
      if (a[j] > val)
	th[j] = a[j] - val;
      else
	th[j] = 0;
    *alpha = 0;
    beta[0] = max(b[0], 0);
    beta[1] = max(b[1], 0);
    return;
  }

  lower = max(0, -b2); // alpha > lower (note: f(0) > 0)
  if (lower == -b2)
    ffold = f(-b2, a, q, b, c, mu);
  for (j = 0; j < q; j++)
    if (th[j] > lower)
      break;
  I = j;
  //printf("I=%d\n", I);
  for (j = I; j < q; j++) {
    ff = f(th[j], a, q, b, c, mu);
    if (ff <= 0) {
      // alpha <= th[j]
      if (ff == 0) {
	// optimal alpha is right at this knot!
	*alpha = th[j];
	for (j = 0; j < q; j++) {
	  if (a[j] > *alpha)
	    th[j] = a[j] - *alpha;
	  else
	    th[j] = 0;
	}
	beta[0] = max(b[0] + *alpha, 0);
	beta[1] = max(b[1] + *alpha, 0);
	//printf("right on knot\n");
	return;
      }
      //printf("alpha < ca[%d]-mu=%f.  f(cur)=%f.  f(prev)=%f\n", j, th[j], ff,ffold);
      if (j == I)
	prev = lower; // lower < alpha <= th[0]
      else
	prev = th[j-1];// th[j-1] < alpha <= th[j]
      cur = th[j];
      if (-b1 > prev && -b1 < th[j]) {
	//printf("-b1 is between\n");
	val = f(-b1, a, q, b, c, mu);
	if (val > 0) {
	  prev = -b1;
	  ffold = val;
	} else if (val < 0) {
	  cur = -b1;
	  ff = val;
	} else {
	  // optimal alpha is -b1!
	  *alpha = -b1;
	  for (j = 0; j < q; j++)
	    th[j] = a[j] - *alpha;
	  beta[0] = max(b[0] + *alpha, 0);
	  beta[1] = max(b[1] + *alpha, 0);
	  return;
	}
      }
      // prev < alpha < cur, and f(alpha) is linear within this range
      *alpha = (prev * ff - cur * ffold) / (ff - ffold); // introduces some numerical error...
      //printf("alpha=%f.  lower=%f, ca_q-mu=%f, -b_1=%f\n",*alpha, lower,th[q-1],-b1);
      val = (mu + *alpha) / c;
      for (j = 0; j < q; j++) {
	if (a[j] > val)
	  th[j] = a[j] - val;
	else
	  th[j] = 0;
      }
      beta[0] = max(b[0] + *alpha, 0);
      beta[1] = max(b[1] + *alpha, 0);
      return;
    }
    ffold = ff;
  }
}

void onerow_withdiag(double *a, int q, int jj, double *b, double c, double mu,
	    double *th, double *beta, double *alpha) {
  // assume a_j >= 0
  // jj is the index of the row we're on.
  int j, I;
  double b1, b2, lower; // b1 <= b2
  double val, ffold, ff, prev, cur;
  for (j = 0; j < q; j++) {
    th[j] = c * a[j] - mu;
  }
  th[jj] -= mu;// diagonal element has param 2mu.
  R_rsort(th, q);
  // check if th is entirely 0:
  if (th[q-1] <= 0) { // max(a) <= c / mu
    // entire row of theta is zero, without help from hierarchy
    for (j = 0; j < q; j++)
      th[j] = 0;
    *alpha = 0;
    beta[0] = max(b[0], 0);
    beta[1] = max(b[1], 0);
    return;
  }
  if (b[0] >= b[1]) {
    b1 = b[1];
    b2 = b[0];
  } else {
    b1 = b[0];
    b2 = b[1];
  }
  if (th[q-1] <= -b2) { // max(a) <= (c - b2) / mu
    // entire row of theta is zero thanks to hierarchy'
    for (j = 0; j < q; j++)
      th[j] = 0;
    *alpha = th[q-1];
    beta[0] = 0;
    beta[1] = 0;
    return;
  }
  // end screening.
  ffold = f_withdiag(0, a, q, jj, b, c, mu);
  //printf("f(0)=%f\n", ffold);
  if (ffold <= 0) {
    // Unconstrained solution is feasible!
    val = mu / c;
    for (j = 0; j < q; j++)
      if (j == jj) {
	if (a[j] > 2*val)
	  th[j] = a[j] - 2*val;
	else
	  th[j] = 0;
      } else {
	if (a[j] > val)
	  th[j] = a[j] - val;
	else
	  th[j] = 0;
      }
    *alpha = 0;
    beta[0] = max(b[0], 0);
    beta[1] = max(b[1], 0);
    return;
  }

  lower = max(0, -b2); // alpha > lower (note: f(0) > 0)
  if (lower == -b2)
    ffold = f_withdiag(-b2, a, q, jj, b, c, mu);
  for (j = 0; j < q; j++)
    if (th[j] > lower)
      break;
  I = j;
  //printf("I=%d\n", I);
  //PrintMatrix(th,1,q);
  for (j = I; j < q; j++) {
    ff = f_withdiag(th[j], a, q, jj, b, c, mu);
    if (ff <= 0) {
      // alpha <= th[j]
      if (ff == 0) {
	// optimal alpha is right at this knot!
	*alpha = th[j];
	for (j = 0; j < q; j++) {
	  if (a[j] > *alpha)
	    th[j] = a[j] - *alpha;
	  else
	    th[j] = 0;
	}
	beta[0] = max(b[0] + *alpha, 0);
	beta[1] = max(b[1] + *alpha, 0);
	//printf("right on knot\n");
	return;
      }
      //printf("alpha < ca[%d]-mu=%f.  f(cur)=%f.  f(prev)=%f\n", j, th[j], ff,ffold);
      if (j == I)
	prev = lower; // lower < alpha <= th[0]
      else
	prev = th[j-1];// th[j-1] < alpha <= th[j]
      cur = th[j];
      if (-b1 > prev && -b1 < th[j]) {
	//printf("-b1 is between\n");
	val = f_withdiag(-b1, a, q, jj, b, c, mu);
	if (val > 0) {
	  prev = -b1;
	  ffold = val;
	} else if (val < 0) {
	  cur = -b1;
	  ff = val;
	} else {
	  // optimal alpha is -b1!
	  *alpha = -b1;
	  for (j = 0; j < q; j++)
	    th[j] = a[j] - *alpha;
	  beta[0] = max(b[0] + *alpha, 0);
	  beta[1] = max(b[1] + *alpha, 0);
	  return;
	}
      }
      // prev < alpha < cur, and f(alpha) is linear within this range
      *alpha = (prev * ff - cur * ffold) / (ff - ffold); // introduces some numerical error...
      //printf("prev * ff - cur * ffold=%f, ff=%f, ffold=%f\n",prev * ff - cur * ffold,ff,ffold);
      //printf("alpha=%f.  lower=%f, ca_q-mu=%f, -b_1=%f\n",*alpha, lower,th[q-1],-b1);
      val = (mu + *alpha) / c;
      for (j = 0; j < q; j++) {
	if (j == jj) {
	  if (a[j] > (2*mu + *alpha) / c)
	    th[j] = a[j] - (2*mu + *alpha) / c;
	  else
	    th[j] = 0;
	} else {
	  if (a[j] > val)
	    th[j] = a[j] - val;
	  else
	    th[j] = 0;
	}
      }
      beta[0] = max(b[0] + *alpha, 0);
      beta[1] = max(b[1] + *alpha, 0);
      return;
    }
    ffold = ff;
  }
}



double f(double alpha, double *a, int q, double *b, double c, double mu) {
  double aa = 0;
  int j;
  double del = (alpha + mu) / c;
  for (j = 0; j < q; j++) {
    if (a[j] > del)
      aa += a[j] - del;
  }
  for (j = 0; j < 2; j++) {
    del = b[j] + alpha;
    if (del > 0)
      aa -= del;
  }
  return aa;
}

double f_withdiag(double alpha, double *a, int q, int jj, double *b, double c, double mu) {
  double aa = 0;
  int j;
  double del = (alpha + mu) / c;
  double del2 = (alpha + 2 * mu) / c;
  //printf(".. alpha=%f, del=%f, del2=%f\n",alpha,del,del2);
  for (j = 0; j < q; j++) {
    if (j == jj) {
      if (a[j] > del2) aa += a[j] - del2;
    } else {
      if (a[j] > del) aa += a[j] - del;
    }
  }
  for (j = 0; j < 2; j++) {
    del = b[j] + alpha;
    if (del > 0)
      aa -= del;
  }
  return aa;
}

void f_R(double *alpha, double *a, int *q, double *b, double *c, double *mu,
	 double *val) {
  *val = f(*alpha, a, *q, b, *c, *mu);
}

void PrintMatrix(double *m, int nrow, int ncol) {
  int i, j;
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++) {
      Rprintf("%g\t", m[i + nrow * j]);
    }
    Rprintf("\n");
  }
}

void ComputeCrossProd(double *m1, int nrow1, int ncol1, 
		      double *m2, int ncol2, double *c) {
  /* Computes m1^T m2.   where m1 is nrow1 by ncol1
                         and m2 is nrow1 by ncol2
     c should be of length ncol1*ncol2*/
  int i, j, k;
  for (j = 0; j < ncol1; j++) {
    for (k = 0; k < ncol2; k++) {
      c[j + ncol1 * k] = 0;
      for (i = 0; i < nrow1; i++)
	c[j + ncol1 * k] += m1[i + nrow1 * j] * m2[i + nrow1 * k];
    }
  }

}

