function run_spectral_test(A, B, σ, tol)
    n = size(A,1)
    nrma = opnorm(A)
    nrmb = opnorm(B)

    F = definite_gen_eigen(A, B, σ, ηx_max=100.0)
    α = F.alphas
    β = F.betas
    λ = F.values
    V = F.vectors

    @testset "Residual test, eigen" begin
        z = [ norm(β[k]*(A * V[:,k]) - α[k] * (B * V[:,k])) /
            (norm(V[:,k]) * (nrma*abs(β[k]) + nrmb * abs(α[k])))
              for k in eachindex(α) ]
        @test norm(z,Inf) <= tol
    end

    @testset "Condition test, eigen" begin
        z = [ 1/cond(β[k] * A - α[k]*B, Inf)
              for k in eachindex(α) ]
        @test norm(z,Inf) <= tol
    end

    ps = definite_gen_eigvals(A, B, σ, ηx_max=100.0)

    @testset "Condition test, eigvals" begin
        z = [ 1/cond(ps[k][2] * A - ps[k][1]*B, Inf)
              for k in eachindex(ps) ]
        @test norm(z,Inf) <= tol
    end
end

function run_interval_test(A, B, σ, tol, vl, vu, ni)
    n = size(A,1)
    nrma = opnorm(A)
    nrmb = opnorm(B)

    F = definite_gen_eigen(A, B, σ, ηx_max=100.0, vl=vl, vu=vu)
    α = F.alphas
    β = F.betas
    λ = F.values
    V = F.vectors

    @testset "Residual test, with interval" begin
        z = [ norm(β[k]*(A * V[:,k]) - α[k] * (B * V[:,k])) /
            (norm(V[:,k]) * (nrma*abs(β[k]) + nrmb * abs(α[k])))
              for k in eachindex(α) ]
        @test norm(z,Inf) <= tol
    end

    @testset "Condition test, eigen, with interval" begin
        z = [ 1/cond(β[k] * A - α[k]*B, Inf)
              for k in eachindex(α) ]
        @test norm(z,Inf) <= tol
    end

    @testset "Number of eigenvalues test, eigen, with interval" begin
        @test length(α) == ni
    end

    ps = definite_gen_eigvals(A, B, σ, ηx_max=100.0, vl = vl, vu = vu)

    @testset "Condition test, eigvals, with interval" begin
        z = [ 1/cond(ps[k][2] * A - ps[k][1]*B, Inf)
              for k in eachindex(ps) ]
        @test norm(z,Inf) <= tol
    end

    @testset "Number of eigenvalues test, eigen, with interval" begin
        @test length(ps) == ni
    end

end

function run_sort_test(A, B, σ, tol, vl, vu)
    n = size(A,1)
    nrma = opnorm(A)
    nrmb = opnorm(B)
    sort_fun((a,b)) = a/b
    F = definite_gen_eigen(A, B, σ, ηx_max=100.0, sortby=sort_fun)

    @testset "Sort test." begin
        issorted(F.values)
    end

    F = definite_gen_eigen(A, B, σ, ηx_max=100.0, sortby=sort_fun, vl=vl, vu=vu)

    @testset "Sort test." begin
        issorted(F.values)
    end
end
