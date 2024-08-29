function shift(
  A::AbstractMatrix{E},
  B::AbstractMatrix{E},
  σ,
) where {E<:AbstractFloat}
  A1 = similar(A)
  for k in axes(A, 2)
    for j in axes(A, 1)
      A1[j, k] = fma(-σ, B[j, k], A[j, k])
    end
  end
  return A1
end

function shift(
  A::AbstractMatrix{Complex{E}},
  B::AbstractMatrix{Complex{E}},
  σ,
) where {E<:AbstractFloat}
  A1 = similar(A)
  for k in axes(A, 2)
    for j in axes(A, 1)
      zr = fma(-σ, real(B[j, k]), real(A[j, k]))
      zi = fma(-σ, imag(B[j, k]), imag(A[j, k]))
      A1[j, k] = complex(zr, zi)
    end
  end
  return A1
end

struct EtaXError{T} <: Exception
  etax :: T
end

function eig_spectral_trans(A, B, σ; ηx_max = 500.0, tol = 0.0)

  Base.require_one_based_indexing(A,B)
  
  m, n = size(A)
  mb, nb = size(B)
  m == n ||
    throw(DimensionMismatch("Matrix A is not square: dimensions are ($m, $n)"))
  mb == nb ||
    throw(DimensionMismatch("Matrix B is not square: dimensions are ($mb,$nb)"))
  n == nb ||
    throw(DimensionMismatch(
      "Matrix A has dimensions ($m,$n) and B has dimensions ($mb,$nb)"))

  Fb = cholesky(Hermitian(B), RowMaximum(), tol = tol, check = false)
  A1 = shift(A, B, σ)

  r = Fb.rank
  ip = invperm(Fb.p)
  Cb = Matrix(Fb.L)[ip, 1:r]
  
  Fa = lqd(Hermitian(A1, :L))

  Da = Fa.S
  η = sqrt(opnorm(A1, Inf) / opnorm(B, Inf))
  
  X = Fa\Cb
  ηx = η * opnorm(X, Inf)
  ηx <= ηx_max || throw(EtaXError(ηx))

  W = X'*(Da*X)

  θ, U = eigen(Hermitian(W))
  λ = similar(θ)
  β = copy(θ)
  α = similar(θ)
  for j in 1:r
    α[j] = fma(σ, θ[j], one(λ[j]))
    λ[j] = α[j]/β[j]
  end
  V = Fa' \ (Da*(X*U))
  return Cb, U, θ, λ, α, β, V, X, η, Da
end
