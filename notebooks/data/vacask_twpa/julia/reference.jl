using JosephsonCircuits, LinearAlgebra, Printf, SHA, TOML, Logging
import Pkg

struct CheckedLogger <: AbstractLogger
    messages::Vector{String}
end
Logging.min_enabled_level(::CheckedLogger) = Logging.Warn
Logging.shouldlog(::CheckedLogger, level, _module, group, id) = true
Logging.catch_exceptions(::CheckedLogger) = false
function Logging.handle_message(
        log::CheckedLogger, level, message, _module, group, id, file, line; kwargs...)
    push!(log.messages, string(message))
    println(stderr, level, ": ", message)
end

const PUMP_FREQUENCY_HZ = 7.12e9

function build_circuit(cells::Int, matched::Bool, ic::Real)
    circuit = Any[("P1", "n0", "0", 1), ("Rs", "n0", "0", 50.0),
        ("P2", "n$cells", "0", 2), ("Rl", "n$cells", "0", 50.0)]
    for i in 0:cells
        ground = i==0 || i==cells ? 22.5e-15 : matched && i%4==1 ? 15e-15 : 45e-15
        push!(circuit, ("Cg$i", "n$i", "0", ground))
        if matched && i>0 && i<cells && i%4==1
            push!(circuit, ("Cc$i", "n$i", "r$i", 30e-15))
            push!(circuit, ("Cr$i", "r$i", "0", 2.8153e-12))
            push!(circuit, ("Lr$i", "r$i", "0", 170e-12))
        end
        if i<cells
            push!(circuit, ("Lj$i", "n$i", "n$(i+1)", JosephsonCircuits.phi0/ic))
            push!(circuit, ("Cj$i", "n$i", "n$(i+1)", 55e-15))
            push!(circuit, ("Rj$i", "n$i", "n$(i+1)", 1e9))
        end
    end
    return circuit
end

function photon_to_voltage(photon, frequency, spur)
    return photon*sqrt(abs(frequency + spur*PUMP_FREQUENCY_HZ)/frequency)
end

function main()
    BLAS.set_num_threads(1)
    cells = parse(Int, get(ENV, "QPDK_JULIA_CELLS", "20"))
    h = parse(Int, get(ENV, "QPDK_JULIA_H", "35"))
    k = parse(Int, get(ENV, "QPDK_JULIA_K", "41"))
    ns = parse(Int, get(ENV, "QPDK_JULIA_SAMPLES", "1609"))
    matched = get(ENV, "QPDK_JULIA_MATCHED", "1") == "1"
    frequencies = parse.(Float64, split(get(ENV, "QPDK_JULIA_FREQUENCIES", "4.5e9,5e9,5.5e9"), ','))
    output = ENV["QPDK_JULIA_OUTPUT"]
    mkpath(output)
    source_tree = bytes2hex(Pkg.GitTools.tree_hash(dirname(dirname(pathof(JosephsonCircuits)))))
    source_tree == "43fa323200866910a93dd06ab8c002ed393ed9bb" ||
        error("Registered Julia solver source changed")
    phi_native = 6.62606876e-34/(4*pi*1.602176462e-19)
    alpha = JosephsonCircuits.phi0/phi_native
    ic, pump = 3.4e-6*alpha, 3.7e-6*alpha
    circuit = build_circuit(cells, matched, ic)

    started = time()
    previous = nothing
    nonlinear = nothing
    stages = Any[]
    for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]
        log = CheckedLogger(String[])
        nonlinear = with_logger(log) do
            hbnlsolve((2*pi*PUMP_FREQUENCY_HZ,), ((ns-1)÷2,),
                [(mode = (1,), port = 1, current = fraction*pump/2)],
                circuit, Dict{Symbol, Float64}(); maxharmonics = (h,),
                dc = false, odd = true, even = false,
                iterations = 200, ftol = 1e-11, x0 = previous, sorting = :name, keyedarrays = true)
        end
        any(occursin("did not converge", lowercase(m)) for m in log.messages) &&
            error("Julia pump did not converge")
        previous = vec(collect(nonlinear.nodeflux))
        push!(stages, Dict("fraction"=>fraction, "warnings"=>log.messages))
        println("Pump fraction ", fraction, " complete")
        flush(stdout)
    end
    nonlinear === nothing && error("No pump orbit")
    open(joinpath(output, "node_phase.csv"), "w") do io
        println(io, "node,mode,ordinary_re,ordinary_im")
        for node in nonlinear.nodes[2:end], mode in nonlinear.frequencies.modes

            value = nonlinear.nodeflux(outputmode = mode, node = node)
            isfinite(value) || error("Nonfinite Julia node phase")
            @printf(io, "%s,%d,%.17g,%.17g\n", node, mode[1], real(value), imag(value))
        end
    end
    println("Linearizing ", length(frequencies), " frequencies")
    flush(stdout)
    linear = hblinsolve(2*pi .* frequencies, circuit, Dict{Symbol, Float64}();
        Nmodulationharmonics = (k,), maxharmonics = (k,), nonlinear = nonlinear,
        sorting = :name, returnS = true, returnSnoise = false,
        returnQE = false, returnCM = false, keyedarrays = true)
    open(joinpath(output, "scattering.csv"), "w") do io
        println(io, "frequency_hz,output_port,input_port,output_spur,re,im")
        for (fi, f) in enumerate(frequencies), op in 1:2, ip in 1:2, spur in [0, -2]
            photon = linear.S(outputmode = (spur,), outputport = op,
                inputmode = (0,), inputport = ip, freqindex = fi)
            value = photon_to_voltage(photon, f, spur)
            isfinite(value) || error("Nonfinite Julia scattering")
            @printf(io, "%.17g,%d,%d,%d,%.17g,%.17g\n", f, op, ip, spur, real(value),
                imag(value))
        end
    end
    metadata = Dict("source_tree_sha1"=>source_tree,
        "package_version"=>string(Base.pkgversion(JosephsonCircuits)),
        "julia_version"=>string(VERSION), "cells"=>cells,
        "phase_matching"=>matched, "pump_harmonics"=>h, "sideband_cutoff"=>k,
        "pump_samples"=>ns, "signal_hz"=>frequencies, "current_scale"=>alpha,
        "phi0_native_reduced"=>phi_native, "phi0_julia_reduced"=>JosephsonCircuits.phi0,
        "nominal_pump_peak_A"=>3.7e-6, "actual_pump_peak_A"=>pump, "pump_stages"=>stages,
        "elapsed"=>time()-started, "script_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    open(joinpath(output, "metadata.toml"), "w") do io
        TOML.print(io, metadata)
    end
    println("Julia reference complete: ", time()-started, " seconds")

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
