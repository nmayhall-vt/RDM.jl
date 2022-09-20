using RDM
using Documenter

DocMeta.setdocmeta!(RDM, :DocTestSetup, :(using RDM); recursive=true)

makedocs(;
    modules=[RDM],
    authors="Nick Mayhall <nmayhall@vt.edu> and contributors",
    repo="https://github.com/nmayhall-vt/RDM.jl/blob/{commit}{path}#{line}",
    sitename="RDM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall-vt.github.io/RDM.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall-vt/RDM.jl",
    devbranch="main",
)
