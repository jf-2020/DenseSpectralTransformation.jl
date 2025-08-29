using LinearAlgebra: HermOrSym, eigtype, eigencopy_oftype

"""
    DefiniteGenEigen <: Factorization

Matrix factorization type of the generalized eigenvalue/spectral
decomposition of `A` and `B`, where `A` and `B` are symmetric or
Hermitian and `B` is positive definite or semidefinite. This is the
return type of [`definite_gen_eigen`](@ref).

For `F::DefiniteGenEigen`, the eigenvalues are stored as a vector
of tuples `F.pairs` and the corresponding eigenvectors as the
columns of the matrix `F.vectors`.

Iterating the decomposition produces the components `F.pairs`, `F.vectors`,
`F.values`, `F.alphas`, and `F.betas`.
"""
struct DefiniteGenEigen{F,V,S<:AbstractMatrix,
                        P<:AbstractVector{Tuple{V, V}}} <: Factorization{F}
    pairs::P
    vectors::S
    DefiniteGenEigen{F,V,S,P}(pairs::AbstractVector{Tuple{V, V}},
                              vectors::AbstractMatrix{F}) where {F,V,S,P} =
                                  new(pairs, vectors)
end

DefiniteGenEigen(pairs::AbstractVector{Tuple{V, V}},
                 vectors::AbstractMatrix{T}) where {T,V} =
    DefiniteGenEigen{T,V,typeof(vectors),typeof(pairs)}(pairs, vectors)

Base.iterate(S::DefiniteGenEigen) = (S.pairs, Val(:vectors))
Base.iterate(S::DefiniteGenEigen, ::Val{:vectors}) = (S.vectors, Val(:values))
Base.iterate(S::DefiniteGenEigen, ::Val{:values}) = (S.values, Val(:alphas))
Base.iterate(S::DefiniteGenEigen, ::Val{:alphas}) = (S.alphas, Val(:betas))
Base.iterate(S::DefiniteGenEigen, ::Val{:betas}) = (S.betas, Val(:done))
Base.iterate(S::DefiniteGenEigen, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::DefiniteGenEigen)
    summary(io, F)
    println(io)
    println(io, "pairs:")
    show(io, mime, F.pairs)
    println(io, "\nvectors:")
    show(io, mime, F.vectors)
end

function Base.getproperty(F::DefiniteGenEigen, sym::Symbol)
    if sym === :alphas
        return first.(getfield(F,:pairs))
    elseif sym === :betas
        return (x -> x[2]).(getfield(F,:pairs))
    elseif sym === :values
        return first.(getfield(F,:pairs)) ./ (x -> x[2]).(getfield(F,:pairs))
    else
        return getfield(F, sym)
    end
end

function shift!(
    A::AbstractMatrix{E},
    B::AbstractMatrix{E},
    σ,
    ) where {E<:AbstractFloat}
    for k in axes(A, 2)
        for j in axes(A, 1)
            A[j, k] = fma(-σ, B[j, k], A[j, k])
        end
    end
    return A
end

function shift!(
    A::AbstractMatrix{Complex{E}},
    B::AbstractMatrix{Complex{E}},
    σ,
    ) where {E<:AbstractFloat}

    for k in axes(A, 2)
        for j in axes(A, 1)
            zr = fma(-σ, real(B[j, k]), real(A[j, k]))
            zi = fma(-σ, imag(B[j, k]), imag(A[j, k]))
            A[j, k] = complex(zr, zi)
        end
    end
    return A
end

struct EtaXError{T} <: Exception
    etax :: T
    bound :: T
end

function triangularize!(A::AbstractMatrix, uplo::Symbol; r = size(A,1))
    z = zero(eltype(A))
    n = size(A,1)
    if uplo === :L
        for k in 2:r
            for j in 1:(k - 1)
                A[j, k] = z
            end
        end
    elseif uplo === :U
        for k in 1:r-1
            for j in k+1:r
                A[j, k] = z
            end
        end
    else
        throw(ArgumentError("uplo must be :L or :U."))
    end
end

@views function sym_mul_lower_blocked!(X,
                                       D;
                                       bs = 64,
                                       work1 = zeros(eltype(X), size(X,1),
                                                     min(bs, size(X,2))),
                                       work2 = zeros(eltype(X), size(X,1),
                                                     min(bs, size(X,2))))

    n, r = size(X)
    work1 = work1[1:n, 1:min(bs, r)]
    work2 = work2[1:n, 1:min(bs, r)]
    lb, remb = divrem(r, bs)
    for l = 1:lb
        @. work1 = X[:, (l-1)*bs + 1 : l*bs]
        @. work2 = X[:, (l-1)*bs + 1 : l*bs]
        lmul!(D, work1[:, 1:bs])
        mul!(X[l*bs+1:r, (l-1)*bs + 1 : l*bs], X[:, l*bs + 1 : r]', work1)
        mul!(X[(l-1)*bs+1:l*bs, (l-1)*bs+1:l*bs], work1[:, 1:bs]', work2)
    end
    if remb > 0
        @. work1[:, 1:remb] = X[:, lb*bs + 1 : r]
        @. work2[:, 1:remb] = X[:, lb*bs + 1 : r]
        lmul!(D, work1[:, 1:remb])
        mul!(X[lb*bs + 1 : r, lb*bs + 1 : r], work1[:, 1:remb]', work2[:, 1:remb])
    end
    return nothing
end

@views function rmul_blocked!(X, U; bs = 64, work = zeros(eltype(X), min(bs, size(X,1)),
                                                          size(X,2)))
    n, r = size(X)
    m = size(U,2)
    bs = min(bs, size(X,1))
    work = work[1:bs, 1:r]
    lb, remb = divrem(n, bs)
    for l = 1:lb
        @. work = X[(l-1)*bs + 1 : l*bs, :]
        mul!(X[(l-1)*bs + 1 : l*bs, 1:m], work, U)
    end
    @. work[1:remb, :] = X[lb * bs + 1 : n, :]
    mul!(X[lb * bs + 1 : n, 1:m], work[1:remb, :], U)
    return nothing
end

function fill_hermitian!(A::RealHermSymComplexHerm)
    Base.require_one_based_indexing(A)

    m = size(A,1)

    if A.uplo == 'U'
        for j in 1:m
            for k in j+1:m
                A.data[k,j] = conj(A.data[j,k])
            end
        end
    elseif A.uplo == 'L'
        for j in 1:m
            for k in j+1:m
                A.data[j,k] = conj(A.data[k,j])
            end
        end
    end

end

function copy_hermitian!(A::RealHermSymComplexHerm, 
                         B::RealHermSymComplexHerm;
                         r = size(A,1))

    Base.require_one_based_indexing(A, B)

    ma = size(A,1)
    mb = size(B,1)

    ma == mb ||
        throw(DimensionMismatch(
            "Matrix A has dimensions ($ma,$ma) and B has dimensions ($mb,$mb)"))

    m = ma

    if A.uplo == 'U'
        if B.uplo == 'U'
            for k in 1:m
                for j in 1:min(k,r)
                    B.data[j,k] = A.data[j,k]
                end
            end
        else
            for k in 1:m
                for j in 1:min(k,r)
                    B.data[k,j] = conj(A.data[j,k])
                end
            end
        end
    elseif A.uplo == 'L'
        if B.uplo == 'U'
            for k in 1:r
                for j in k:m
                    B.data[k,j] = conj(A.data[j,k])
                end
            end
        else
            for k in 1:r
                for j in k:m
                    B.data[j,k] = A.data[j,k]
                end
            end
        end
    end

end

function norm_est(A; tol=0.05, maxiters=100)

    m, n = size(A)
    v = similar(A, n)
    u = similar(A,m)
    v .= vec(mapreduce(abs, +, A, dims = 1))
    normv = norm(v)
    v = v/normv
    normA = normv
    normA0 = zero(normv)
    iters = 0
    At = A'
    while abs(normA0 - normA) > tol*normA && iters < maxiters
        normA0 = normA
        mul!(u, A, v)
        mul!(v, At, u)
        normA = norm(v)
        v .= v ./ normA
        iters += 1
    end
    iters >= maxiters && (@warn "Maximum iterations reached in norm_est")
    return sqrt(normA), iters
end

# The following computes a Hermitian matrix W associated with the
# shifted and inverted generalized eigenvalue problem.  It also
# returns the rank of B, established by tolerance tol, along with
# factorizations of A and B.  However, the lower triangular pivoted
# Cholesky factorization of B is in the upper triangular part of B.
# Fb does include the permutation.
@views function _setup_eig!(
    A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    B::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    σ::Real;
    ηx_max = 500.0,
    tol = 0.0,
    bs = 64,
    vl = nothing,
    vu = nothing,
    bound_norm_est = true,
    throw_bound_error = false
    )

    Base.require_one_based_indexing(A, B)
    n, nb = LinearAlgebra.checksquare(A,B)
    n == nb ||
        throw(DimensionMismatch(
            "Matrix A has dimensions ($n,$n) and B has dimensions ($nb,$nb)"))

    fill_hermitian!(A)
    fill_hermitian!(B)

    A = A.data
    B = B.data

    E = promote_type(eltype(A), eltype(B), eltype(σ))
    tmpa = Array{E}(undef, n)
    tmpb = Array{E}(undef, n)

    shift!(A, B, σ)
    η = bound_norm_est ? sqrt(norm_est(A)[1] / norm_est(B)[1]) :
        sqrt(opnorm(A, Inf) / opnorm(B, Inf))


    Fb = cholesky!(Hermitian(B, :L), RowMaximum(), tol = tol, check = false)

    r = Fb.rank
    ip = invperm(Fb.p)

    Cb = Fb.factors
    triangularize!(Cb, :L; r=r)

    Fa = lqd!(Hermitian(A, :L))

    Da = Fa.S
    # Do X = Fa\Cb
    copy_hermitian!(Hermitian(Cb, :L), Hermitian(A, :U); r = r)

    X = Cb[:, 1:r]
    Base.permutecols!!(X', copy(ip))
    Base.permutecols!!(X', copy(Fa.p))
    ldiv!(Fa.L, X)
    ldiv_LQD_Q!(Fa.Q, X)
    ldiv!(Fa.D, X)

    ηx = bound_norm_est ? η * norm_est(X)[1] : η * opnorm(X, Inf)
    if ηx > ηx_max
        if throw_bound_error
            ηx <= ηx_max || throw(EtaXError(ηx, ηx_max))
        else
            @warn "Value of η ||X|| is $(ηx), which exceeds the given bound $(ηx_max)."
        end
    end


    sym_mul_lower_blocked!(X, Da; bs = bs)
    W = Hermitian(X[1:r, 1:r], :L)

    return r, Fa, Fb, W

end

@views function definite_gen_eigen!(
    A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    B::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    σ::Real;
    ηx_max = 500.0,
    tol = 0.0,
    bs = 64,
    vl = nothing,
    vu = nothing,
    bound_norm_est = true,
    throw_bound_error = false,
    sortby::Union{Nothing, Function}=nothing
    )

    r, Fa, Fb, W = _setup_eig!(A, B, σ; ηx_max = ηx_max, tol = tol, bs = bs,
                               bound_norm_est = bound_norm_est,
                               throw_bound_error = throw_bound_error)

    θ, U = _eigen_interval!(W, σ; vl=vl, vu=vu)
    m = length(θ)

    λ = similar(θ)
    β = copy(θ)
    α = similar(θ)
    for j in 1:m
        α[j] = fma(σ, θ[j], one(λ[j]))
        λ[j] = α[j] / β[j]
    end

    A = A.data
    B = B.data
    copy_hermitian!(Hermitian(A, :U), Hermitian(B, :L); r = r)
    X = B[:,1:r]
    triangularize!(X, :L; r =r)

    # recompute X
    Base.permutecols!!(X', invperm(Fb.p))
    Base.permutecols!!(X', copy(Fa.p))
    ldiv!(Fa.L, X)
    ldiv_LQD_Q!(Fa.Q, X)
    ldiv!(Fa.D, X)


    # Compute V = Fa' \ (Da*(X*U))
    rmul_blocked!(X, U, bs = bs)
    V = X[:, 1:m]
    lmul!(Fa.S, V)
    ldiv!(Fa.D, V)
    lmul_LQD_Q!(Fa.Q, V)
    ldiv!(Fa.L', V)
    Base.permutecols!!(V', invperm(Fa.p))
    pairs = collect(zip(α, β))
    return DefiniteGenEigen(LinearAlgebra.sorteig!(pairs, V, sortby)...)
end

"""
    definite_gen_eigen(A::Hermitian, B::Hermitian, σ::Real;
                       ηx_max = 500.0, tol = 0.0,
                       bs = 64, vl = nothing, vu = nothing,
                       bound_norm_est = true, throw_bound_error = false,
                       sortby::Union{Function, Nothing}=nothing) -> DefiniteGenEigen

Compute the generalized eigenvalues of Hermitian `A` and `B`, with `B`
assumed positive definite or semidefinite, using a shift and invert
spectral transformation with shift σ.  A [`DefiniteGenEigen`](@ref) is
returned.  For `F::DefiniteGenEigen`, the eigenvalues are stored as a
vector of tuples `F.pairs` and the corresponding eigenvectors as the
columns of the matrix `F.vectors`.  Iterating the decomposition
produces the components `F.pairs`, `F.vectors`, `F.values`,
`F.alphas`, and `F.betas`.  The pairs are of the form `F.pairs[k] ==
(F.alphas[k], F.betas[k])` and the generalized eigenvalues are
`F.values[k] == F.alphas[k] / F.betas[k]`.

The matrix `B` is assumed to be positive definite or semidefinite.  A
(possibly truncated) pivoted Cholesky factorization PᵀBP = LLᵀ is
computed and the routine computes as an intermediate result the
eigenvalues and eigenvectors of `Lᵀ Pᵀ (A - σ B)⁻¹ P L`.  These are
then used to compute the finite generalized eigenvalues and eigenvectors of A
and B.  That is, for each `k`
```
F.beta[k] * A * F.vectors[:,k] ≈ F.alpha[k] * B * F.vectors[:,k]
```
and
```
A * F.vectors[:,k] ≈ F.values[k] * B * F.vectors[:,k]
```

The null space of `B` gives eigenvectors for infinite generalized
eigenvalues.  This is currently not computed.  It should be noted that
if `B` is not positive definite, then there is no guarantee that there
are `n` linearly independent generalized eigenvectors.  The way that
this manifests is that `V` will include a generalized eigenvector
that is also in the null space of `B`.

The stability properties are not entirely simple, but are better in
most cases than those of the standard algorithm that works with an
eigenvalue decomposition `L⁻ᵀ A L⁻¹` and exhibits large residuals for
smaller eigenvalues.  The algorithm can be used in one of two ways: If
`σ` is chosen to be not too close to a generalized eigenvalue and if
the normalized shift `σ₀ = σ ||B||/||A||` is of moderate size, then it
can be shown that the spectral transformation algorithm gives
generalized eigenvalues that all achieve small residuals for some
choice of eigenvector (but not necessarily for the computed eigenvector).
For eigenvalues that are not dramatically larger than `σ`, the
computed eigenvectors give a small residual.  Alternately, if a large
scaled shift `σ₀` is used, then computed eigenvectors for eigenvalues
not too far from the shift will achieve small residuals.

Keyword parameters:

- `ηx_max = 500`: An upper bound on a scaled norm of an intermediate
  matrix `X` that appears in a factor in the error bounds.  This
  factor is large when `σ` is chosen to be too close to a generalized
  eigenvalue.  If the computed quantity exceeds this bound, either warn
  the user or throw `EtaXError`, depending on `throw_bound_error`.

- `bound_norm_est = true`: Use power iteration to estimate the 2-norm of `X` before
  applying `ηx_max` as a threshold.  Otherwise the ∞-norm is used, which is typically
  more conservative than needed for stability.

- `throw_bound_error = false`:  If `false`, exceeding `ηx_max` is just a warning.

- `tol=0`: The truncation tolerance for a truncated pivoted Cholesky.
  If `B` is positive definite, the default `tol=0` is appropriate and a full
  Cholesky decomposition is computed.  

- `bs = 64`: Block size for blocked multiplies of lower and upper triangular
  matrices.  Adjusting this might change run times for the better or worse.

- `vl = nothing, vu = nothing`: Upper and lower bounds on eigenvalues.

- `sortby=nothing`: Nothing or a function to apply prior to sorting of eigenvalues.
"""
function definite_gen_eigen(A::HermOrSym{TA}, B::HermOrSym{TB}, σ::TS;
                            ηx_max = 500.0, tol = 0.0,
                            bs = 64, vl = nothing, vu = nothing,
                            bound_norm_est = true, throw_bound_error = false,
                            sortby::Union{Function, Nothing}=nothing) where {TA,
                                                                             TB,
                                                                             TS <: Real}
    T = promote_type(eigtype(TA), TB, TS)
    return definite_gen_eigen!(eigencopy_oftype(A,T), eigencopy_oftype(B,T),
                               convert(real(T), σ), ηx_max = ηx_max, tol=tol,
                               bs = bs, vl = vl, vu = vu,
                               bound_norm_est = bound_norm_est,
                               throw_bound_error = throw_bound_error,
                               sortby=sortby)
end

@views function definite_gen_eigvals!(
    A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    B::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    σ::Real;
    ηx_max = 500.0,
    tol = 0.0,
    bs = 64,
    vl = nothing,
    vu = nothing,
    bound_norm_est = true,
    throw_bound_error = false,
    sortby::Union{Nothing, Function}=nothing
    )

    r, Fa, Fb, W = _setup_eig!(A, B, σ; ηx_max = ηx_max, tol = tol, bs = bs,
                               bound_norm_est = bound_norm_est,
                               throw_bound_error = throw_bound_error)

    θ = _eigvals_interval!(W, σ; vl=vl, vu=vu)
    m = length(θ)

    λ = similar(θ)
    β = copy(θ)
    α = similar(θ)
    for j in 1:m
        α[j] = fma(σ, θ[j], one(λ[j]))
        λ[j] = α[j] / β[j]
    end
    return LinearAlgebra.sorteig!(collect(zip(α, β)), sortby)
end

function definite_gen_eigvals(A, B, σ; ηx_max = 500.0, tol = 0.0,
                              bs = 64, vl = nothing, vu = nothing,
                              bound_norm_est = true, throw_bound_error = false,
                              sortby::Union{Function, Nothing}=nothing)
    return definite_gen_eigvals!(copy(A), copy(B), σ, ηx_max = ηx_max, tol=tol,
                                 bs = bs, vl = vl, vu = vu,
                                 bound_norm_est = bound_norm_est,
                                 throw_bound_error = throw_bound_error,
                                 sortby=sortby)
end
