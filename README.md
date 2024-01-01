# DenseSpectralTransformation.jl

This package applies the shift and invert strategy from the spectral transformation Lanczos method for the symmetric semidefinite generalized eigenvalue problem to dense problems.  This approach is based on computing eigenvalues and vectors of $C_b^T (A-\sigma B)^{-1} C_b$ where $B = C_b C_b^T$ is a pivoted Cholesky factor of $B$, possibly with rank truncation if $B$ is semidefinite.  The code is currently proof of concept code for a paper that is under review.  The paper investigates the stability of this as a method for dense problems and includes some relatively strong bounds on residuals if $\sigma$ is chosen suitably.  The call for solving $Ax = \lambda Bx$ where $A$ and $B$ are hermitian and $B$ is semidefinite is
```
using DenseSpectralTransformations
Cb, U, θ, λ, α, β, V, X, η, Da =
  eig_spectral_trans(A, B, σ; ηx_max = 500.0, tol = 0.0)
```
The computed eigenvalues are in the vector $\lambda$ and the corresponding eigenvectors in the matrix $V$.  The vectors $\alpha$ and $\beta$ are the components of eigenvalues represented as pairs $(\alpha_i, \beta_i)$ such that $\beta_i A v_i = \alpha_i B v_i$.  Thus $\lambda_i = \alpha_i/\beta_i$.  The stability results in the paper focus on eigenvalues in this form.  The eigenvalues and eigenvectors of the shifted and inverted problem are in $\theta$ and $U$.  The parameter `tol` determines what is neglected in truncating the factorization of $B$.  A tolerance of $0.0$ attempts to compute a full pivoted Cholesky factorization.  The parameter `ηx_max` is a bound on a quantity $\eta ||X||_2$ related to the stability of the algorithm.  If the computed value of $\eta ||X||_2$ exceeds this threshold, the algorithm fails.  The size of $\eta ||X||_2$ depends on the shift.  For success, $\sigma$ should not be too close to an eigenvalue in a relative sense and the scaled shift $\sigma_0 =\sigma ||B||_2/||A||_2$ should not be too large.  If $A$ is positive definite, simply choosing $\sigma_0 = -2$ and computing $\sigma$ from $\sigma_0$, possibly using a norm other than the 2-norm, should give good eigenvalues, although the quality of the eigenvectors is not guaranteed for eigenvalues much larger in magnitude than $\sigma$.  There are additional details on the choice of shift in the paper.

## Numerical Experiments

The code includes a subproject in the directory `experiments` for the numerical experiments presented in the paper.  The experiments run the code on matrices that are part of the Harwell-Boeing collection from the NIST [MatrixMarket](https://math.nist.gov/MatrixMarket/) website.  The commit used to generate the graphs given in the submitted paper is tagged as `PaperSubmitted`.  The `Manifest.toml` files for the project and subprojects are not tracked.  However, in the specific tagged commit, I have included a file `Manifests.zip` that will reproduce the manifests in the proper locations when unzipped.  After doing this, the numerical experiments should run from within the `experiments` directory with
```
Pkg.activate(".")
Pkg.instantiate()
include("paper_experiments.jl")
```

## Efficiency

The current implementation of the algorithm favors simplicity over efficiency.  The bulk of the computation happens in Julia routines that call out to LAPACK.  But I didn't implement anything in-place and the code does allocate new matrices for everything that is computed.  This is something I plan to change in future versions.
