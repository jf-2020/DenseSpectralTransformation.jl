using DenseSpectralTransformation
using SafeTestsets
using LinearAlgebra
using Test

@safetestset "LQDS Tests Lower, Float64" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_lqds.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    A = Float64[
        (i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1])
        for i = 1:n, j = 1:n
            ]
    tol = 1e-13
    run_lqds_test(Hermitian(A, :L), tol)
end

@safetestset "LQDS Tests Upper, Float64" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_lqds.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    A = Float64[
        (i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1])
        for i = 1:n, j = 1:n
            ]
    tol = 1e-13
    run_lqds_test(Hermitian(A, :U), tol)
end

@safetestset "LQDS Tests Lower, Complex{Float64}" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_lqds.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    y = exp.(-.25 * (n:-1:1))
    A = Complex{Float64}[
        i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1] + im*y[abs(i-j)+1]
        for i = 1:n, j = 1:n
            ]
    tol = 1e-13
    run_lqds_test(Hermitian(A, :L), tol)
end

@safetestset "LQDS Tests Upper, Complex{Float64}" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_lqds.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    y = exp.(-.25 * (n:-1:1))
    A = Complex{Float64}[
        i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1] + im*y[abs(i-j)+1]
        for i = 1:n, j = 1:n
            ]
    tol = 1e-13
    run_lqds_test(Hermitian(A, :U), tol)
end


@safetestset "Eigenvalue/vector Test, Float64" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_spectral.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    A = Float64[
        (i == j ? (-1)^i * x[abs(i - j) + 1] : x[abs(i - j) + 1])
        for i = 1:n, j = 1:n
            ]
    B = [1/(i+j-1) for i in 1:n, j in 1:n]
    tol = 1e-13
    run_spectral_test(Hermitian(A), Hermitian(B), 5.0, tol)
end

@safetestset "Eigenvalue/vector Test, Complex{Float64}" begin
    using LinearAlgebra
    using DenseSpectralTransformation
    include("../test/test_spectral.jl")
    n = 300
    x = exp.(-.2 * (1:n))
    y = exp.(-.25 * (n:-1:1))
    A = zeros(Complex{Float64}, n, n)
    for i in 1:n
        for j in 1:i
            A[i,j] = 
                i == j ? (-1)^i * x[abs(i - j) + 1] :
                x[abs(i - j) + 1] + im*y[abs(i-j)+1]
        end
    end
    A=A+A'
    B = Complex{Float64}[1/(i+j-1) for i in 1:n, j in 1:n]
    tol = 1e-13
    run_spectral_test(Hermitian(A), Hermitian(B), 5.0, tol)
end

