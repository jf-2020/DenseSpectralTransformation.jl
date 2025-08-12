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

    @testset "Left inverse" begin
        @test opnorm(inv(X) - F\I0) <= tol * sqrt(nrma)
    end

end
