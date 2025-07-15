using LinearAlgebra

anyIsInf(a) = any(x -> abs(x)==Inf, a)

function inv_norm_est(U::UpperTriangular{E}; tol=1e-15, maxits=10) where E
    m, n = size(U)
    x = randn(E, n)
    x .= x ./ norm(x)
    dx = one(real(E))
    y = zeros(E,n)
    z = zeros(E,n)
    z0 = zeros(E,n)
    x0 = zeros(E,n)
    for j ∈ 1:n
        iszero(U[j,j]) && return Inf
    end
    count = 0
    ymax = 0
    while count < maxits
        count = count + 1
        ldiv!(y, U, x)
        ymax = anyIsInf(y) ? (return Inf) : maximum(abs, y)
        y .= y ./ ymax
        ldiv!(z, U', y)
        zmax = anyIsInf(z) ? (return Inf) : maximum(abs, z)
        z0 .= z ./ norm(z)
        dx = abs(one(E) - abs(x⋅z0))
        dx < tol && break
        x .= z0
    end
    return sqrt(ymax*norm(z))
end
