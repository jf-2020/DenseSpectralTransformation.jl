using LinearAlgebra
using DenseSpectralTransformation
using JET

function run_lqds()
  
  Ar = Hermitian(randn(n, n), :L)
  F = lqd(Ar)
  B = randn(n,n)
  X = Matrix(F)
  F*B
  B*F
  F\B
  B/F
  F'*B
  B*F'
  F'\B
  B/F'

  Ac = Hermitian(randn(Complex{Float64}, n, n), :L)
  F = lqd(Ac)
  B = randn(Complex{Float64}, n, n)
  X = Matrix(F)
  F*B
  B*F
  F\B
  B/F
  F'*B
  B*F'
  F'\B
  B/F'
end

function run_spectral()
  n = 5
  tol = 1e-13
  Br = [1 / (i + j - 1) for i = 1:n, j = 1:n]
  Ar = randn(n, n)
  Ar = Ar + Ar'
  eig_spectral_trans(Ar, Br, 5.0, ηx_max=100.0)

  Bc = Complex{Float64}[1 / (i + j - 1) for i = 1:n, j = 1:n]
  Ac = randn(Complex{Float64}, n, n)
  Ac = Ac + Ac'
  eig_spectral_trans(Ac, Bc, 5.0, ηx_max=100.0)
end

println("LQDS @report_opt:")
display(@report_opt target_modules = (DenseSpectralTransformation,) run_lqds())

println("LQDS @report_call:")
display(@report_call target_modules = (DenseSpectralTransformation,) run_lqds())

println("Spectral @report_opt:")
display(@report_opt target_modules = (DenseSpectralTransformation,) run_spectral())

println("Spectral @report_call:")
display(@report_call target_modules = (DenseSpectralTransformation,) run_spectral())
