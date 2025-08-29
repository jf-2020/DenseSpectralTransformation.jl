# DenseSpectralTransformation.jl

This package applies the shift and invert strategy from the spectral transformation Lanczos method for the symmetric semidefinite generalized eigenvalue problem to dense problems.  This approach is based on computing eigenvalues and vectors of $C_b^T (A-\sigma B)^{-1} C_b$ where $B = C_b C_b^T$ is a pivoted Cholesky factor of $B$, possibly with rank truncation if $B$ is semidefinite.  The code is currently proof of concept code for a paper that is under review.  The paper investigates the stability of this as a method for dense problems and includes some relatively strong bounds on residuals if $\sigma$ is chosen suitably.  The primary function for solving $Ax = \lambda Bx$ where $A$ and $B$ are hermitian and $B$ is semidefinite is
```
    F = definite_gen_eigen(A::Hermitian, B::Hermitian, σ::Real;
                           ηx_max = 500.0, tol = 0.0,
                           bs = 64, vl = nothing, vu = nothing,
                           bound_norm_est = true, throw_bound_error = false,
                           sortby::Union{Function, Nothing}=nothing) -> DefiniteGenEigen
```
This computes the generalized eigenvalues of Hermitian $A$ and $B$, with $B$
assumed positive definite or semidefinite, using a shift and invert
spectral transformation with shift $\sigma$.  A `DefiniteGenEigen` is
returned.  For `F::DefiniteGenEigen`, the eigenvalues are stored as a
vector of tuples `F.pairs` and the corresponding eigenvectors as the
columns of the matrix `F.vectors`.  Iterating the decomposition
produces the components `F.pairs`, `F.vectors`, `F.values`,
`F.alphas`, and `F.betas`.  The pairs are of the form `F.pairs[k] ==
(F.alphas[k], F.betas[k])` and the generalized eigenvalues are
`F.values[k] == F.alphas[k] / F.betas[k]`.

The matrix $B$ is assumed to be positive definite or semidefinite.  A (possibly truncated) pivoted Cholesky factorization $P^T B P = L L^T$ is computed and the routine computes as an intermediate result the eigenvalues and eigenvectors of $L^T P^T (A-\sigma B)^{-1} P L$.  These are then used to compute the finite generalized eigenvalues and eigenvectors of $A$ and $B$.  That is, for each $k$
```
F.beta[k] * A * F.vectors[:,k] ≈ F.alpha[k] * B * F.vectors[:,k]
```
and
```
A * F.vectors[:,k] ≈ F.values[k] * B * F.vectors[:,k]
```
The null space of $B$ gives eigenvectors for infinite generalized
eigenvalues.  This is currently not computed.  It should be noted that
if $B$ is not positive definite, then there is no guarantee that there
are $n$ linearly independent generalized eigenvectors.  The way that
this manifests is that $V$ will include a generalized eigenvector
that is also in the null space of $B$.

The stability properties are not entirely simple, but are better in most cases than those of the standard algorithm that works with an eigenvalue decomposition $L^{-T} A L^{-1}$ and exhibits large residuals for smaller eigenvalues.  The algorithm can be used in one of two ways: If $\sigma$ is chosen to be not too close to a generalized eigenvalue and if the normalized shift $\sigma_0 = \sigma ||B||/||A||$ is of moderate size, then it can be shown that the spectral transformation algorithm gives generalized eigenvalues that all achieve small residuals for some choice of eigenvector (but not necessarily for the computed eigenvector).  For eigenvalues that are not dramatically larger than $\sigma$, the computed eigenvectors give a small residual.  Alternately, if a large scaled shift $\sigma_0$ is used, then computed eigenvectors for eigenvalues not too far from the shift will achieve small residuals.

## Numerical Experiments

The code includes a subproject in the directory `experiments` for the numerical experiments presented in the paper.  The experiments run the code on matrices that are part of the Harwell-Boeing collection from the NIST [MatrixMarket](https://math.nist.gov/MatrixMarket/) website.  The commit used to generate the graphs given in the submitted paper is tagged as `PaperSubmitted`.  The `Manifest.toml` files for the project and subprojects are not tracked.  However, in the specific tagged commit, I have included a file `Manifests.zip` that will reproduce the manifests in the proper locations when unzipped.  After doing this, the numerical experiments should run from within the `experiments` directory with
```
Pkg.activate(".")
Pkg.instantiate()
include("paper_experiments.jl")
```

## Updates

The current implementation of the algorithm works with matrices in place where possible and should be more efficient than the tagged version.  The experiments should still work.
