using QCBase
using RDM
using TensorOperations
using LinearAlgebra

"""
    compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T

Return energy defined by spin-summed  `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, d1::ssRDM1{T}, d2::ssRDM2{T}) where T
#={{{=#
    length(d1.rdm) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * d1.rdm[p,q]
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.rdm[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T

Return energy defined by `rdm1` and `rdm2`.
# Arguments
- `ints`: InCoreInts object
- `d1`:   1 particle reduced density matrix
- `d2`:   2 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, d1::RDM1{T}, d2::RDM2{T}) where T
#={{{=#
    length(d1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(d1.b) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(d1)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (d1.a[p,q] + d1.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * d2.aa[p,q,r,s]
        e += .5 * ints.h2[p,q,r,s] * d2.bb[p,q,r,s]
        e +=      ints.h2[p,q,r,s] * d2.ab[p,q,r,s]
    end
    
    return e
end
#=}}}=#

"""
    compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T

Return energy defined by `rdm1`.
# Arguments
- `ints`: InCoreInts object
- `rdm1`: 1 particle reduced density matrix
"""
function QCBase.compute_energy(ints::InCoreInts{T}, rdm1::RDM1{T}) where T
#={{{=#
    length(rdm1.a) == length(ints.h1) || throw(DimensionMismatch)
    length(rdm1.b) == length(ints.h1) || throw(DimensionMismatch)

    no = n_orb(ints)
    e = ints.h0
    
    for p in 1:no, q in 1:no
        e += ints.h1[p,q] * (rdm1.a[p,q] + rdm1.b[p,q])
    end
    
    for p in 1:no, q in 1:no, r in 1:no, s in 1:no
        e += .5 * ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.a[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.a[p,s] * rdm1.a[r,q]
        
        e += .5 * ints.h2[p,q,r,s] * rdm1.b[p,q] * rdm1.b[r,s]
        e -= .5 * ints.h2[p,q,r,s] * rdm1.b[p,s] * rdm1.b[r,q]
        
        e += ints.h2[p,q,r,s] * rdm1.a[p,q] * rdm1.b[r,s]
    end
    
    return e
end
#=}}}=#


"""
    compute_fock(ints::InCoreInts, rdm1::RDM1)

Compute Fock Matrix
"""
function compute_fock(ints::InCoreInts, rdm1::RDM1)
#={{{=#
    fa = deepcopy(ints.h1)
    fb = deepcopy(ints.h1)
    @tensor begin
        #a
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q] 
        fa[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.a[p,q]
        fa[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        
        #b
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.b[p,q] 
        fb[r,s] -= 0.5 * ints.h2[p,r,q,s] * rdm1.b[p,q]
        fb[r,s] += 0.5 * ints.h2[p,q,r,s] * rdm1.a[p,q]
        
    end
    return (fa,fb) 
end
#=}}}=#



function LinearAlgebra.tr(d::RDM1)
    return tr(d.a)+tr(d.b)
end
function LinearAlgebra.tr(d::RDM2)
    n = n_orb(d) 
    t = 0.0
    for p in 1:n
        for q in 1:n
            t += d.aa[p,p,q,q]
            t += d.ab[p,p,q,q]
            t += d.ab[q,q,p,p]
            t += d.bb[p,p,q,q]
        end
    end
    return t
end


"""
    subset(ints::InCoreInts, list, rmd1a, rdm1b)
Extract a subset of integrals acting on orbitals in list, returned as `InCoreInts` type
and contract a 1rdm to give effectve 1 body interaction
# Arguments
- `ints::InCoreInts`: Integrals for full system 
- `list`: list of orbital indices in subset
- `rdm1a`: 1RDM for embedding α density to make CASCI hamiltonian
- `rdm1b`: 1RDM for embedding β density to make CASCI hamiltonian
"""
function InCoreIntegrals.subset(ints::InCoreInts, ci::MOCluster, rdm1::RDM1)
    list = ci.orb_list
    ints_i = subset(ints, list)
    da = deepcopy(rdm1.a)
    db = deepcopy(rdm1.b)
    da[:,list] .= 0
    db[:,list] .= 0
    da[list,:] .= 0
    db[list,:] .= 0
    viirs = ints.h2[list, list,:,:]
    viqri = ints.h2[list, :, :, list]
    f = zeros(length(list),length(list))
    @tensor begin
        f[p,q] += viirs[p,q,r,s] * (da+db)[r,s]
        f[p,s] -= .5*viqri[p,q,r,s] * da[q,r]
        f[p,s] -= .5*viqri[p,q,r,s] * db[q,r]
    end
    ints_i.h1 .+= f
    #h0 = compute_energy(ints, RDM1(da,db))
    return InCoreInts(0.0, ints_i.h1, ints_i.h2) 
end
