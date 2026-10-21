import Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

function main(paths)
    failed = false
    for path in paths
        try
            # format_text() bypasses the ancestor config search format(path) does.
            source = read(path, String)
            formatted = format_text(source, SciMLStyle(); remove_extra_newlines = false)
            if formatted != source
                write(path, formatted)
                failed = true
            end
        catch err
            println(stderr, "$(path): JuliaFormatter failed: ", sprint(showerror, err))
            failed = true
        end
    end
    return failed
end

exit(main(ARGS) ? 1 : 0)
