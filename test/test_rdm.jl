using QCBase
using RDM  
using Test
using ActiveSpaceSolvers
using NPZ
using Printf
using InCoreIntegrals
using JLD2
using LinearAlgebra
using Random

function generate_test_data()
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")
    
    ints = InCoreInts(h0, h1, h2)

    ansatz = FCIAnsatz(6,3,3)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)
    e_ref = solution.energies[1]
    @save "h6_sto3g/rdms_33.jld2" e_ref ints da db daa dbb dab


    
    ansatz = FCIAnsatz(6,4,2)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    e_ref = solution.energies[1]
    
    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)

    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)

    @test isapprox(compute_energy(ints, d1, d2), solution.energies[1])
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), solution.energies[1])

    @save "h6_sto3g/rdms_42.jld2" e_ref ints da db daa dbb dab
end

@testset "RDM" begin
    @load "h6_sto3g/rdms_33.jld2" 
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    @test isapprox(compute_energy(ints, d1, d2), e_ref)
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), e_ref)
    
    display(norm(d1.a - RDM1(d2).a))
    display(norm(d1.b - RDM1(d2).b))
    @test isapprox(norm(d1.a - RDM1(d2).a),0,atol=1e-14)
    @test isapprox(norm(d1.b - RDM1(d2).b),0,atol=1e-14)

    fa, fb = RDM.compute_fock(ints, d1)
    e,C = eigen(fa+fb)
    display(e)
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
    
    g = build_orbital_gradient(ints, d1, d2)
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
   
    println()
    println()
    @load "h6_sto3g/rdms_42.jld2" 
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    @test isapprox(compute_energy(ints, d1, d2), e_ref)
    @test isapprox(compute_energy(ints, ssRDM1(d1), ssRDM2(d2)), e_ref)
    
    @test isapprox(norm(d1.a - RDM1(d2).a),0,atol=1e-14)
    @test isapprox(norm(d1.b - RDM1(d2).b),0,atol=1e-14)
    @test isapprox(tr(d1.a), 4, atol=1e-14)
    @test isapprox(tr(d1.b), 2, atol=1e-14)
    println()
    fa, fb = RDM.compute_fock(ints, d1)
    e,C = eigen(fa+fb)
    display(e)

    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
    
    g = build_orbital_gradient(ints, d1, d2)
    @printf("\n Orbital Gradient should be zero\n")
    display(norm(g))
    @test isapprox(norm(g),0.0,atol=1e-7)
end


function numgrad()
    h0 = npzread("h6_sto3g/h0.npy")
    h1 = npzread("h6_sto3g/h1.npy")
    h2 = npzread("h6_sto3g/h2.npy")
    
    ints = InCoreInts(h0, h1, h2)

    ansatz = FCIAnsatz(6,3,2)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=100)
    solution = ActiveSpaceSolvers.solve(ints, ansatz, solver)
    display(solution)
    Random.seed!(1)
    v = rand(size(solution.vectors)...)

    v = v/norm(v)
    solution.vectors .= v

    rdm1a, rdm1b = compute_1rdm(solution)
    da, db, daa, dbb, dab = compute_1rdm_2rdm(solution)
    d1 = RDM1(da,db)
    d2 = RDM2(daa, dab, dbb)
    e1 = compute_energy(ints, d1, d2)
    e2 = compute_energy(ints, ssRDM1(d1), ssRDM2(d2))

    @printf(" %12.8f\n", e1)
    @printf(" %12.8f\n", e2)

    no = n_orb(ints)
    k = RDM.pack_gradient(zeros(no,no), no)
    grad = deepcopy(k)

    stepsize=1e-6
    for i in 1:length(k)
        ki = deepcopy(k)
        ki[i] += stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e1 = compute_energy(intsi, d1, d2)
        
        ki = deepcopy(k)
        ki[i] -= stepsize
        Ki = RDM.unpack_gradient(ki, no)
        U = exp(Ki)
        intsi = orbital_rotation(ints,U)
        e2 = compute_energy(intsi, d1, d2)
        
        grad[i] = (e1-e2)/(2*stepsize)
    end
    println(" Numerical Gradient: ")
    display(grad)
    
    println(" Analytical Gradient: ")
    g = build_orbital_gradient(ints, d1, d2)
    display(g)

    println()
    display(norm(g-grad))
    g = build_orbital_gradient(ints, ssRDM1(d1), ssRDM2(d2))
    display(g)

    println()
    display(norm(g-grad))
end
