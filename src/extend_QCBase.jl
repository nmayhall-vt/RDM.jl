using QCBase
using RDM 
using TensorOperations
using LinearAlgebra


QCBase.n_orb(r::RDM1) = size(r.a,1)
QCBase.n_orb(r::RDM2) = size(r.aa,1)
QCBase.n_orb(c::Cumulant2) = size(c.a,1)
QCBase.n_orb(r::ssRDM1) = size(r.rdm,1)
QCBase.n_orb(r::ssRDM2) = size(r.rdm,1)




