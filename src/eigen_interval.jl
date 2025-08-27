function _eigen_interval!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
                          σ::Real; vl = nothing, vu = nothing)
    n = size(A,1)
    if isnothing(vl) && isnothing(vu)
        return eigen!(A)
    else
        vl = isnothing(vl) ? -Inf : vl
        vu = isnothing(vu) ? Inf : vu
        if vl <= σ <= vu
            throw(ArgumentError(lazy"Shift must be outside [vl, vu]."))
        else
            θu = 1/(vl - σ)
            θl = 1/(vu - σ)
        end
        F = eigen!(A, θl, θu)
        @static if VERSION >= v"1.12-"
            if length(F.values) < n
                return Eigen(copy(F.values), copy(F.vectors))
            end
        end
        return F
    end
end

function _eigvals_interval!(A::RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix},
                            σ::Real; vl = nothing, vu = nothing)
    n = size(A,1)
    if isnothing(vl) && isnothing(vu)
        return eigvals!(A)
    else
        vl = isnothing(vl) ? -Inf : vl
        vu = isnothing(vu) ? Inf : vu
        if vl <= σ <= vu
            throw(ArgumentError(lazy"Shift must be outside [vl, vu]."))
        else
            θu = 1/(vl - σ)
            θl = 1/(vu - σ)
        end
        e = eigvals!(A, θl, θu)
        @static if VERSION >= v"1.12-"
            if length(e) < n
                return copy(e)
            end
        end
        return e
    end
end
