using QCBase
using RDM 
using TensorOperations
using LinearAlgebra

"""
    a::Array{T,2}   # α
    b::Array{T,2}   # β

1 particle RDM type 
"""
struct phRDM1{T} <: AbstractArray{T,2} 
    a::Dict{String, Array{T,2}}
    b::Dict{String, Array{T,2}}
end

# """
#     aa::Array{T,2}  # αα
#     ab::Array{T,2}  # αβ 
#     bb::Array{T,2}  # ββ

# 2 particle RDM type 
# """
# struct RDM2{T} <: AbstractArray{T,4} 
#     aa::Array{T,4}
#     ab::Array{T,4}
#     bb::Array{T,4}
# end