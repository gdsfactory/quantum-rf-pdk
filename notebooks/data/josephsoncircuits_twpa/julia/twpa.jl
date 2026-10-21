using JosephsonCircuits

const PUMP_FREQUENCY_HZ = 7.12e9
const CRITICAL_CURRENT_A = 3.4e-6
# JosephsonCircuits port sources take half the cosine peak of the Norton current.
const PUMP_SOURCE_A = 3.7e-6 / 2

"""
    build_circuit(cells, matched, ic=CRITICAL_CURRENT_A)

Josephson ladder of `cells` RCSJ junctions between two 50 Ω ports, after the
resonantly phase-matched JTWPA of O'Brien et al. (2014) as distributed with
JosephsonCircuits.jl. With `matched`, every fourth cell couples to an LC
resonator at 7.275 GHz.
"""
function build_circuit(cells::Int, matched::Bool, ic::Real = CRITICAL_CURRENT_A)
    circuit = Any[("P1", "n0", "0", 1), ("Rs", "n0", "0", 50.0),
        ("P2", "n$cells", "0", 2), ("Rl", "n$cells", "0", 50.0)]
    for i in 0:cells
        ground = i == 0 || i == cells ? 22.5e-15 : matched && i % 4 == 1 ? 15e-15 : 45e-15
        push!(circuit, ("Cg$i", "n$i", "0", ground))
        if matched && i > 0 && i < cells && i % 4 == 1
            push!(circuit, ("Cc$i", "n$i", "r$i", 30e-15))
            push!(circuit, ("Cr$i", "r$i", "0", 2.8153e-12))
            push!(circuit, ("Lr$i", "r$i", "0", 170e-12))
        end
        if i < cells
            push!(circuit, ("Lj$i", "n$i", "n$(i+1)", JosephsonCircuits.phi0 / ic))
            push!(circuit, ("Cj$i", "n$i", "n$(i+1)", 55e-15))
            push!(circuit, ("Rj$i", "n$i", "n$(i+1)", 1e9))
        end
    end
    return circuit
end

"""
    solve_twpa(frequencies, cells, matched; pump_harmonics=20, modulation_harmonics=10)

Pump the ladder at `PUMP_FREQUENCY_HZ` and linearize it at the signal
`frequencies` (Hz), returning the JosephsonCircuits solution with S-parameters,
quantum efficiency and the commutation-relation check.
"""
function solve_twpa(frequencies, cells::Int, matched::Bool;
        pump_harmonics::Int = 20, modulation_harmonics::Int = 10,
        pump_current::Real = PUMP_SOURCE_A)
    sources = [(mode = (1,), port = 1, current = pump_current)]
    return hbsolve(2pi .* frequencies, (2pi * PUMP_FREQUENCY_HZ,), sources,
        (modulation_harmonics,), (pump_harmonics,), build_circuit(cells, matched),
        Dict{Symbol, Float64}(); returnS = true, returnQE = true, returnCM = true,
        sorting = :name)
end
