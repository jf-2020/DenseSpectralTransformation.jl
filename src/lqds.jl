struct LQD{
    E,
    T<:AbstractMatrix{E},
    V<:AbstractVector{E},
    VI<:AbstractVector{Int64},
    } <: Factorization{E}
    LorUdata::T
    Qdata::T
    ipiv::VI
    uplo::Char
    d::V
    s::VI
    p::VI

    function LQD(
        LorUdata::T,
        Qdata::T,
        ipiv::VI,
        uplo::Char,
        d::V,
        s::VI,
        p::VI,
        ) where {E,T<:AbstractMatrix{E},V<:AbstractVector{E},VI<:AbstractVector} 

        Base.require_one_based_indexing(LorUdata, Qdata)
        new{E,T,V,VI}(LorUdata, Qdata, ipiv, uplo, d, s, p)
    end
end

struct LQD_Q{
    E,
    T<:AbstractMatrix{E},
    VI<:AbstractVector{Int64},
    }
    Qdata::T
    ipiv::VI

    function LQD_Q(
        Qdata::T,
        ipiv::VI,
        ) where {E,T<:AbstractMatrix{E},VI<:AbstractVector} 

        Base.require_one_based_indexing(Qdata)
        new{E,T,VI}(Qdata, ipiv)
    end
end

function lqd!(A::Hermitian{E, <:AbstractMatrix{E}}) where {E}
    F = bunchkaufman!(A, true)
    n, _ = size(A)
    s = similar(F.p)
    uplo = F.uplo == 'U' ? :U : :L
    Qdata = similar(F.LD, n, 2)
    Qdata .= zero(eltype(Qdata))
    d = similar(F.LD, n)
    j = 1
    while j <= n
        if F.ipiv[j] >= 1
            Qdata[j, 1] = one(eltype(Qdata))
            d[j] = sqrt(abs(F.LD[j, j]))
            s[j] = sign(F.LD[j, j])
            j += 1
        else
            dj, Qj = eigen(Hermitian(view(F.LD, j:(j + 1), j:(j + 1)), uplo))
            d[j:(j + 1)] .= sqrt.(abs.(dj))
            s[j:(j + 1)] .= sign.(dj)
            Qdata[j:(j + 1), :] .= Qj
            j += 2
        end
    end

    # TODO: Handle non Blas types.
    LorUdata, _ = LAPACK.syconvf_rook!(F.uplo, 'C', F.LD, F.ipiv)

    return LQD(LorUdata, Qdata, F.ipiv, F.uplo, d, s, F.p)
end

lqd(A::Hermitian{E,<:AbstractMatrix{E}}) where {E} = lqd!(copy(A))

function Base.getproperty(F::LQD, sym::Symbol)
    if sym === :D
        return Diagonal(getfield(F, :d))
    elseif sym === :L
        getfield(F, :uplo) == 'U' &&
            throw(ArgumentError("factorization is U*Q*D*S*D*Q'*U' but you requested L."))
        return UnitLowerTriangular(getfield(F, :LorUdata))
    elseif sym === :U
        getfield(F, :uplo) == 'L' &&
            throw(ArgumentError("factorization is L*Q*D*S*D*Q'*L' but you requested U."))
        return UnitUpperTriangular(getfield(F, :LorUdata))
    elseif sym === :S
        return Diagonal(getfield(F, :s))
    elseif sym === :Q
        LQD_Q(getfield(F, :Qdata), getfield(F, :ipiv))
    else
        return getfield(F, sym)
    end
end

Base.size(lqd::LQD) = size(lqd.LorUdata)
Base.size(lqd::Adjoint{<:LQD}) = size(lqd.parent.LorUdata)

function Base.show(io::IO, mime::MIME"text/plain", lqd::LQD)
    summary(io, lqd)
    println(io)
    println(io, "$(lqd.uplo) factor:")
    show(io, mime, lqd.uplo == 'L' ? lqd.L : lqd.U)
    println(io, "\nQ factor:")
    show(io, mime, lqd.Q)
    println(io, "\nD factor:")
    show(io, mime, lqd.D)
    println(io, "\nS factor:")
    show(io, mime, lqd.S)
end

function Base.:*(
    lqd::LQD{E},
    A::AbstractVecOrMat{E},
    ) where {E}
    return ((lqd.uplo == 'L' ? lqd.L : lqd.U) * (lqd.Q * (lqd.D * A)))[
        invperm(lqd.p),
        :,
    ]
end

function Base.:\(
    lqd::LQD{E},
    A::AbstractVecOrMat{E},
    ) where {E}
    return lqd.D \ (lqd.Q \ ((lqd.uplo == 'L' ? lqd.L : lqd.U) \ A[lqd.p, :]))
end


function Base.Matrix(F::LQD{E}) where E
    n, _ = size(F)
    return F * Matrix{E}(I, n, n)
end

function Base.:*(
    Q::LQD_Q{E},
    A::AbstractMatrix{E},
    ) where {E}
    lmul_LQD_Q!(Q, copy(A))
    return A
end

function Base.:*(
    Q::LQD_Q{E},
    A::AbstractVector{E},
    ) where {E}
    v = reshape(copy(A), length(A),1)
    lmul_LQD_Q!(Q, v)
    return vec(v)
end

function Base.:\(
    Q::LQD_Q{E},
    A::AbstractMatrix{E},
    ) where {E}
    ldiv_LQD_Q!(Q, copy(A))
    return A
end

function Base.:\(
    Q::LQD_Q{E},
    A::AbstractVector{E},
    ) where {E}
    v = reshape(copy(A), length(A),1)
    ldiv_LQD_Q!(Q, v)
    return vec(v)
end

function lmul_LQD_Q!(F::LQD_Q, A::AbstractMatrix)
    Qdata = getfield(F, :Qdata)
    ipiv = getfield(F, :ipiv)
    n = length(ipiv)
    j = 1
    while j <= n
        if ipiv[j] >= 1
            q11 = Qdata[j,1]
            for k in axes(A,2)
                A[j, k] = q11*A[j, k]
            end
            j += 1
        else
            q11 = Qdata[j,1]
            q12 = Qdata[j,2]
            q21 = Qdata[j+1, 1]
            q22 = Qdata[j+1, 2]
            for k in axes(A,2)
                tmp = A[j,k]
                A[j, k] = q11 * A[j, k] + q12 * A[j + 1, k]
                A[j + 1, k] = q21 * tmp + q22 * A[j + 1, k]
            end
            j += 2
        end
    end
    return A
end

function ldiv_LQD_Q!(F::LQD_Q, A::AbstractMatrix)
    Qdata = getfield(F, :Qdata)
    ipiv = getfield(F, :ipiv)
    n = length(ipiv)
    j = 1
    while j <= n
        if ipiv[j] >= 1
            q11 = Qdata[j,1]
            for k in axes(A,2)
                A[j, k] = conj(q11)*A[j, k]
            end
            j += 1
        else
            q11 = Qdata[j,1]
            q12 = Qdata[j,2]
            q21 = Qdata[j+1, 1]
            q22 = Qdata[j+1, 2]
            for k in axes(A,2)
                tmp = A[j,k]
                A[j, k] = conj(q11) * A[j, k] + conj(q21) * A[j + 1, k]
                A[j + 1, k] = conj(q12) * tmp + conj(q22) * A[j + 1, k]
            end
            j += 2
        end
    end
    return A
end
