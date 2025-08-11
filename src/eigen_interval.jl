using LinearAlgebra.LAPACK: @chkvalidparam, @blasfunc, chkuplo

using LinearAlgebra: libblastrampoline, BlasFloat, BlasInt,
    LAPACKException, DimensionMismatch, SingularException, PosDefException,
    chkstride1, checksquare, triu, tril, dot

function eigen_interval!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
                         σ::BlasReal;
                         sortby::Union{Function,Nothing}=nothing,
                         vl = nothing, vu = nothing)
    if isnothing(vl) && isnothing(vu)
        return Eigen(LinearAlgebra.sorteig!(
            syevr_interval!('V', 'A', A.uplo, A.data, 
                            0.0, 0.0, 0, 0, -1.0)..., sortby)...)
    else
        vl = isnothing(vl) ? -Inf : vl
        vu = isnothing(vu) ? Inf : vu
        if vl <= σ <= vu
            throw(ArgumentError(lazy"Shift must be outside [vl, vu]."))
        else
            θu = 1/(vl - σ)
            θl = 1/(vu - σ)
        end
        return Eigen(LinearAlgebra.sorteig!(
            syevr_interval!('V', 'V', A.uplo, A.data, 
                            θl, θu, 0, 0, -1.0)..., sortby)...)
    end
end


for (syevr, elty) in
    ((:dsyevr_,:Float64),
     (:ssyevr_,:Float32))
    @eval begin
        function syevr_interval!(jobz::AbstractChar, range::AbstractChar,
                                 uplo::AbstractChar, A::AbstractMatrix{$elty},
                                 vl::AbstractFloat, vu::AbstractFloat,
                                 il::Integer, iu::Integer, abstol::AbstractFloat)
            Base.require_one_based_indexing(A)
            @chkvalidparam 1 jobz ('N', 'V')
            @chkvalidparam 2 range ('A', 'V', 'I')
            LinearAlgebra.chkstride1(A)
            n = LinearAlgebra.checksquare(A)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError(lazy"illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError(lazy"lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            LAPACK.chkuplofinite(A, uplo)
            lda = stride(A,2)
            m = Ref{BlasInt}()
            W = similar(A, $elty, n)
            ldz = n
            if jobz == 'N'
                Z = similar(A, $elty, ldz, 0)
            elseif jobz == 'V'
                Z = similar(A, $elty, ldz, n)
            end
            isuppz = similar(A, BlasInt, 2*n)
            work   = Vector{$elty}(undef, 1)
            lwork  = BlasInt(-1)
            iwork  = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info   = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ref{BlasInt}, Clong, Clong, Clong),
                      jobz, range, uplo, n,
                      A, max(1,lda), vl, vu,
                      il, iu, abstol, m,
                      W, Z, max(1,ldz), isuppz,
                      work, lwork, iwork, liwork,
                      info, 1, 1, 1)
                LAPACK.chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            end
            return m[] == n ? (W, Z) : (W[1:m[]], Z[:,1:(jobz == 'V' ? m[] : 0)])
        end
    end
end

for (syevr, elty, relty) in
    ((:zheevr_,:ComplexF64,:Float64),
     (:cheevr_,:ComplexF32,:Float32))
    @eval begin
        function syevr_interval!(jobz::AbstractChar, range::AbstractChar,
                                 uplo::AbstractChar, A::AbstractMatrix{$elty},
                                 vl::AbstractFloat, vu::AbstractFloat,
                                 il::Integer, iu::Integer, abstol::AbstractFloat)
            Base.require_one_based_indexing(A)
            @chkvalidparam 1 jobz ('N', 'V')
            @chkvalidparam 2 range ('A', 'V', 'I')
            chkstride1(A)
            LAPACK.chkuplofinite(A, uplo)
            n = LinearAlgebra.checksquare(A)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError(lazy"illegal choice of eigenvalue indices (il = $il, iu=$iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError(lazy"lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            lda = max(1,stride(A,2))
            m = Ref{BlasInt}()
            W = similar(A, $relty, n)
            if jobz == 'N'
                ldz = 1
                Z = similar(A, $elty, ldz, 0)
            elseif jobz == 'V'
                ldz = n
                Z = similar(A, $elty, ldz, n)
            end
            isuppz = similar(A, BlasInt, 2*n)
            work   = Vector{$elty}(undef, 1)
            lwork  = BlasInt(-1)
            rwork  = Vector{$relty}(undef, 1)
            lrwork = BlasInt(-1)
            iwork  = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info   = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1], lrwork as rwork[1] and liwork as iwork[1]
                ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                       Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                       Clong, Clong, Clong),
                      jobz, range, uplo, n,
                      A, lda, vl, vu,
                      il, iu, abstol, m,
                      W, Z, ldz, isuppz,
                      work, lwork, rwork, lrwork,
                      iwork, liwork, info,
                      1, 1, 1)
                LAPACK.chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    lrwork = BlasInt(rwork[1])
                    resize!(rwork, lrwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            end
            return m[] == n ? (W, Z) : (W[1:m[]], Z[:,1:(jobz == 'V' ? m[] : 0)])
        end
    end
end
