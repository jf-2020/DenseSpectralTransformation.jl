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

function shift(A::AbstractMatrix, B::AbstractMatrix, σ)
    A1 = copy(A)
    shift!(A1, B, σ)
    return A1
end

struct EtaXError{T} <: Exception
    etax :: T
end

function save_diag!(A, x)
    n = minimum(size(A))
    for j in 1:n
        x[j] = A[j,j]
    end
end

function restore_diag!(A, x)
    n = minimum(size(A))
    for j in 1:n
        A[j,j] = x[j]
    end
end

# Form the product X'*D*X in W.
@views function sym_mul2!(W, X, D; work = zeros(eltype(X), size(X, 1)))
    n, r = size(X)
    work = work[1:n]
    for k in 1:r
        @. work = X[:, k]
        lmul!(D, work)
        A = W.data
        if W.uplo == 'L'
            for j in k:r
                A[j, k] = zero(eltype(W))
                for l in 1:n
                    A[j, k] += conj(X[l, j]) * work[l]
                end
            end
        elseif W.uplo == 'U'
            for j in 1:k
                A[j, k] = zero(eltype(W))
                for l in 1:n
                    A[j, k] += conj(X[l, j]) * work[l]
                end
            end
        else
            error("uplo should be L or U, not $(W.uplo)")
        end
    end
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
    else
        error("Incorrect value of uplo for a Hermitian matrix.")
    end

end


function eig_spectral_trans!(
    A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    B::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
    σ;
    ηx_max = 500.0,
    tol = 0.0,
    )

    Base.require_one_based_indexing(A, B)

    Ah = A
    Bh = B

    fill_hermitian!(Ah)
    fill_hermitian!(Bh)

    A = Ah.data
    B = Bh.data

    m, n = size(A)
    mb, nb = size(B)

    m == n ||
        throw(DimensionMismatch("Matrix A is not square: dimensions are ($m, $n)"))
    mb == nb ||
        throw(DimensionMismatch("Matrix B is not square: dimensions are ($mb,$nb)"))
    n == nb ||
        throw(DimensionMismatch(
            "Matrix A has dimensions ($m,$n) and B has dimensions ($mb,$nb)"))

    E = promote_type(eltype(A), eltype(B), eltype(σ))
    tmp1 = Array{E}(undef, n)
    tmp2 = Array{E}(undef, n)

    shift!(A, B, σ)

    Fb = cholesky!(Hermitian(B, :L), RowMaximum(), tol = tol, check = false)

    r = Fb.rank
    ip = invperm(Fb.p)

    Cb = view(Fb.factors, :, 1:r)
    z = zero(eltype(Cb))
    for k in 2:r
        for j in 1:(k - 1)
            Cb[j, k] = z
        end
    end

    Base.permutecols!!(Cb', ip)
    η = sqrt(opnorm(A, Inf) / opnorm(B, Inf))

    # Fa = lqd!(Hermitian(A, :U))
    Fa = lqd!(Hermitian(A, :L))

    Da = Fa.S
    # X = Fa\Cb
    X = Cb
    Base.permutecols!!(X', copy(Fa.p))
    ldiv!(Fa.L, X)
    ldiv_LQD_Q!(Fa, X)
    ldiv!(Fa.D, X)

    ηx = η * opnorm(X, Inf)
    ηx <= ηx_max || throw(EtaXError(ηx))

    # W = X'*(Da*X)
    save_diag!(A, tmp2)
    W = Hermitian(view(A, 1:r, 1:r), :U)
    sym_mul2!(W, X, Da; work = tmp1)

    θ, U = eigen!(W)

    λ = similar(θ)
    β = copy(θ)
    α = similar(θ)
    for j in 1:r
        α[j] = fma(σ, θ[j], one(λ[j]))
        λ[j] = α[j] / β[j]
    end
    restore_diag!(A, tmp2)

    # V = Fa' \ (Da*(X*U))
    lmul!(Da, X)
    ldiv!(Fa.D, X)
    lmul_LQD_Q!(Fa, X)
    ldiv!(Fa.L', X)
    Base.permutecols!!(X', invperm(Fa.p))
    V = view(A, :, 1:r)
    mul!(V, X, U)
    return Cb, U, θ, λ, α, β, V, X, η, Da
end

function eig_spectral_trans(A, B, σ; ηx_max = 500.0, tol = 0.0)
    return eig_spectral_trans!(copy(A), copy(B), σ, ηx_max = ηx_max, tol=tol)
end
