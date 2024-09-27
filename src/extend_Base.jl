using RDM 
using TensorOperations
using LinearAlgebra


Base.size(r::RDM1) = return size(r.a)
Base.size(r::RDM2) = return size(r.aa)



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


