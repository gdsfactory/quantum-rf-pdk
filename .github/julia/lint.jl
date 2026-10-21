import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaWorkspaces
using JuliaWorkspaces.URIs2

function add_path!(workspace, path, workspace_path = path)
    uri = filepath2uri(abspath(workspace_path))
    language = endswith(path, ".toml") ? "toml" : "julia"
    add_file!(workspace, TextFile(uri, SourceText(read(path, String), language)))
    return uri
end

function report(workspace, path, uri)
    source = get_text_file(workspace, uri).content
    failed = false
    for diagnostic in get_diagnostic(workspace, uri)
        offset = clamp(first(diagnostic.range), 1, lastindex(source.content) + 1)
        position = position_at(source, offset)
        code = diagnostic.code === nothing ? "" : " [$(diagnostic.code)]"
        println(stderr, "$(path):$(position.line):$(position.column): ",
            diagnostic.severity, ": ", diagnostic.message, code)
        failed |= diagnostic.severity === :error
    end
    return failed
end

function main(paths)
    workspace = JuliaWorkspace()
    config = joinpath(@__DIR__, "JuliaLint.toml")
    root_config = joinpath(dirname(dirname(@__DIR__)), "JuliaLint.toml")
    files = unique([config; abspath.(paths)])
    # Scope the settings across the workspace without creating a root config file.
    uris = [add_path!(workspace, path, path == config ? root_config : path)
            for path in files]

    failed = false
    for (path, uri) in zip(files, uris)
        try
            failed |= report(workspace, path, uri)
        catch err
            println(stderr, "$(path): JuliaWorkspaces failed: ", sprint(showerror, err))
            failed = true
        end
    end
    return failed
end

exit(main(ARGS) ? 1 : 0)
