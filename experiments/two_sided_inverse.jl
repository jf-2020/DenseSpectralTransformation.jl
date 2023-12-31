# Compute L⁻¹AL⁻ᵀ
function two_sided_inv(A, L)
  Base.require_one_based_indexing(A, L)
  C = similar(A)
  n = size(A, 1)
  fill!(C, zero(eltype(A)))
  C[1, 1] = A[1, 1] / (L[1, 1] * L[1, 1])
  @views for k = 2:n
    C[1:(k - 1), k] = LowerTriangular(L[1:(k - 1), 1:(k - 1)]) \ A[1:(k - 1), k]
    x = L[k, 1:(k - 1)] ⋅ C[1:(k - 1), k]
    C[1:(k - 1), k] =
      C[1:(k - 1), k] - C[1:(k - 1), 1:(k - 1)] * L[k, 1:(k - 1)]
    C[k, k] = A[k, k] - x - L[k, 1:(k - 1)] ⋅ C[1:(k - 1), k]
    C[k, k] = C[k, k] / (L[k, k] * L[k, k])
    C[1:(k - 1), k] = C[1:(k - 1), k] / L[k, k]
    C[k, 1:(k - 1)] = C[1:(k - 1), k]
  end
  return C
end

