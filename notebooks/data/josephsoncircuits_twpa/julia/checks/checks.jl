import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Test
using JET

include(joinpath(@__DIR__, "..", "twpa.jl"))

component(circuit, name) = only(filter(entry -> entry[1] == name, circuit))
has_component(circuit, name) = any(entry -> entry[1] == name, circuit)

@testset "circuit construction" begin
    cells = 8
    ic = 3.4e-6
    matched = build_circuit(cells, true, ic)
    unmatched = build_circuit(cells, false, ic)

    @test component(matched, "P1") == ("P1", "n0", "0", 1)
    @test component(matched, "P2") == ("P2", "n8", "0", 2)
    @test component(matched, "Rs")[4] == 50.0
    @test component(matched, "Rl")[2] == "n8"

    @test component(matched, "Cg0")[4] == 22.5e-15
    @test component(matched, "Cg8")[4] == 22.5e-15
    @test component(matched, "Cg1")[4] == 15e-15
    @test component(matched, "Cg2")[4] == 45e-15
    @test component(unmatched, "Cg1")[4] == 45e-15

    @test component(matched, "Cc1") == ("Cc1", "n1", "r1", 30e-15)
    @test component(matched, "Cr1")[4] == 2.8153e-12
    @test component(matched, "Lr1")[4] == 170e-12
    @test component(matched, "Cc5")[2] == "n5"
    @test !has_component(matched, "Cc2")
    @test !has_component(unmatched, "Cc1")
    @test length(matched) == length(unmatched) + 6

    for i in 0:(cells - 1)
        @test component(matched, "Lj$i")[2:3] == ("n$i", "n$(i+1)")
    end
    @test component(matched, "Lj0")[4] == JosephsonCircuits.phi0 / ic
    @test component(build_circuit(2, false, 2ic), "Lj0")[4] ≈ JosephsonCircuits.phi0 / (2ic)
end

@testset "photon balance on a short chain" begin
    frequencies = [5e9, 6e9]
    lin = solve_twpa(frequencies, 64, true; pump_harmonics = 8, modulation_harmonics = 4).linearized
    modes = [m[1] for m in lin.modes]
    for (n, f) in enumerate(frequencies)
        balance = sum(sign(f + k * PUMP_FREQUENCY_HZ) * abs2(lin.S((k,), port, (0,), 1, n))
        for k in modes, port in 1:2)
        @test balance ≈ 1 atol = 1e-6
        @test 0 < lin.QE((0,), 2, (0,), 1, n) / lin.QEideal((0,), 2, (0,), 1, n) <= 1 + 1e-9
    end
end

@testset "JET" begin
    JET.test_call(build_circuit, (Int, Bool, Float64); target_modules = (Main,))
end
