function run_lqds_test(A, tol)
    n = size(A,1)
    nrma = opnorm(A)
    F = lqd(A)
    I0 = Matrix{eltype(A)}(I, n, n)
    X = Matrix(F)

    @testset "Backward error" begin
        @test opnorm(X * (F.S * X') - A) <= tol * nrma
    end

    @testset "Left multiply" begin
        @test opnorm(X - F*I0) <= tol * sqrt(nrma)
    end

    @testset "Right multiply" begin
        @test opnorm(X - I0*F) <= tol * sqrt(nrma)
    end

    @testset "Left inverse" begin
        @test opnorm(inv(X) - F\I0) <= tol * sqrt(nrma)
    end

    @testset "Right inverse" begin
        @test opnorm(inv(X) - I0/F) <= tol * sqrt(nrma)
    end

    @testset "Left adjoint multiply" begin
        @test opnorm(X' - F'*I0) <= tol * sqrt(nrma)
    end

    @testset "Right adjoint multiply" begin
        @test opnorm(X' - I0*F') <= tol * sqrt(nrma)
    end

    @testset "Left adjoint inverse" begin
        @test opnorm(inv(X)' - F'\I0) <= tol * sqrt(nrma)
    end

    @testset "Right adjoint inverse" begin
        @test opnorm(inv(X)' - I0/F') <= tol * sqrt(nrma)
    end

end
