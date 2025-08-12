module DenseSpectralTransformation

using LinearAlgebra
using LinearAlgebra: BlasReal, RealHermSymComplexHerm

include("eigen_interval.jl")

include("lqds.jl")
export lqd!, lqd, LQD, LQD_Q

include("spectral.jl")
export definite_gen_eigen, definite_gen_eigen!,
    shift, EtaXError, DefiniteGenEigen, norm_est
using PrecompileTools

@setup_workload begin
    @compile_workload begin
        n = 5
        tol = 1e-13
        Br = [1 / (i + j - 1) for i = 1:n, j = 1:n]
        Ar = randn(n, n)
        Ar = Ar + Ar'
        definite_gen_eigen(Hermitian(Ar), Hermitian(Br), 5.0, ηx_max=100.0)

        Bc = Complex{Float64}[1 / (i + j - 1) for i = 1:n, j = 1:n]
        Ac = randn(Complex{Float64}, n, n)
        Ac = Ac + Ac'
        definite_gen_eigen(Hermitian(Ac), Hermitian(Bc), 5.0, ηx_max=100.0)
    end
end

end # module SymDefGenEig
