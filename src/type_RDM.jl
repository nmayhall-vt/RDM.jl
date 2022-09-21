using QCBase
using RDM 
using TensorOperations
using LinearAlgebra

struct RDM1{T} <: AbstractArray{T,2} 
    a::Array{T,2}
    b::Array{T,2}
    #RDM1(a::Matrix{T}, b::Matrix{T}) where T = new{T}(deepcopy(a), deepcopy(b))
    #RDM1{T}(a, b) where T = new{T}(deepcopy(a), deepcopy(b))
end
struct RDM2{T} <: AbstractArray{T,4} 
    aa::Array{T,4}
    ab::Array{T,4}
    bb::Array{T,4}
    #RDM2(a::Array{T,4}, b::Array{T,4}, c::Array{T,4}) where T = new{T}(deepcopy(a), deepcopy(b), deepcopy(c))
    #RDM2{T}(a, b, c) where T = new{T}(deepcopy(a), deepcopy(b), deepcopy(c))
end
struct Cumulant2{T} <: AbstractArray{T,4} 
    aa::Array{T,4}
    ab::Array{T,4}
    bb::Array{T,4}
end

# spin-summed RDMs
struct ssRDM1{T} <: AbstractArray{T,2} 
    rdm::Array{T,2}
    #ssRDM1(a::Matrix{T}) where T = new{T}(deepcopy(a))
    #ssRDM1{T}(a) where T = new{T}(deepcopy(a))
end
struct ssRDM2{T} <: AbstractArray{T,4} 
    rdm::Array{T,4}
    #ssRDM2(a::Array{T,4}) where T = new{T}(deepcopy(a))
    #ssRDM2{T}(a) where T = new{T}(deepcopy(a))
end


Base.size(r::RDM1) = return size(r.a)
Base.size(r::RDM2) = return size(r.aa)


function ssRDM1(rdm::RDM1{T}) where T
    return ssRDM1{T}(rdm.a .+ rdm.b)
end
function ssRDM2(rdm::RDM2{T}) where T
    return ssRDM2{T}(rdm.aa .+ rdm.bb .+ rdm.ab + permutedims(rdm.ab, [3,4,1,2]))
end

function RDM2(rdm::RDM1{T}) where T
    no = n_orb(rdm) 
    Daa = zeros(no, no, no, no)
    Dab = zeros(no, no, no, no)
    Dbb = zeros(no, no, no, no)
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        Daa[p,q,r,s] = rdm.a[p,q] * rdm.a[r,s] - rdm.a[p,s] * rdm.a[r,q]
        Dbb[p,q,r,s] = rdm.b[p,q] * rdm.b[r,s] - rdm.b[p,s] * rdm.b[r,q]
        Dab[p,q,r,s] = rdm.a[p,q] * rdm.b[r,s]
    end
    return RDM2{T}(Daa, Dab, Dbb)
end

function QCBase.n_orb(r::RDM1) 
    size(r.a,1)
end
QCBase.n_orb(r::RDM2) = size(r.aa,1)
QCBase.n_orb(c::Cumulant2) = size(c.a,1)

QCBase.n_orb(r::ssRDM1) = size(r.rdm,1)
QCBase.n_orb(r::ssRDM2) = size(r.rdm,1)



"""
    RDM1(rdm::RDM2)

Integrate a 2rdm to get the 1rdm.
We assume that d is stored, d[1,1,2,2]
such that <p'q'rs> is D[p,s,q,r]
Also, we will agree with pyscf, and use the following normalization:
tr(D) = N(N-1)
"""
function RDM1(d2::RDM2{T}) where T
    n = n_orb(d2) 
    d1a = zeros(n,n)
    d1b = zeros(n,n)
    for p in 1:n
        for q in 1:n
            for r in 1:n
                d1a[p,q] += d2.aa[p,q,r,r]
                #d1a[p,q] += d2.ab[p,q,r,r]
                d1b[p,q] += d2.bb[p,q,r,r]
                #d1b[p,q] += d2.ab[r,r,p,q]
            end
        end
    end
    ca = tr(d1a)
    cb = tr(d1b)
    Na = (1 + sqrt(1+4*ca) )/2
    Nb = (1 + sqrt(1+4*cb) )/2
    return RDM1(d1a ./ (Na-1), d1b ./ (Nb-1))
end

Base.:-(da::RDM1, db::RDM1) = return RDM1(da.a.-db.a, da.b.-db.b) 
Base.:+(da::RDM1, db::RDM1) = return RDM1(da.a.+db.a, da.b.+db.b) 

function Base.display(da::ssRDM1)
    display("RDM1")
    display(da.rdm)
end

function Base.display(da::RDM1)
    display("RDM1(α)")
    display(da.a)
    display("RDM1(β)")
    display(da.b)
end

function Base.display(d::RDM2)
    @printf("RDM2(αα) size: %s\n", size(d.aa))
    @printf("RDM2(αβ) size: %s\n", size(d.ab))
    @printf("RDM2(ββ) size: %s\n", size(d.bb))
end


