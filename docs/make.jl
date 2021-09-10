using Diffractor
using Documenter
using DocThemeIndigo
using Markdown

DocMeta.setdocmeta!(
    Diffractor,
    :DocTestSetup,
    :(using Diffractor),
)

indigo = DocThemeIndigo.install(Diffractor)

makedocs(
    modules=[Diffractor],
    format=Documenter.HTML(
        prettyurls=false,
        assets=[indigo],
        mathengine=MathJax3(
            Dict(
                :tex => Dict(
                    "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                    "tags" => "ams",
                    # TODO: remove when using physics package
                    "macros" => Dict(
                        "ip" => ["{\\left\\langle #1, #2 \\right\\rangle}", 2],
                        "Re" => "{\\operatorname{Re}}",
                        "Im" => "{\\operatorname{Im}}",
                        "tr" => "{\\operatorname{tr}}",
                    ),
                ),
            ),
        ),
    ),
    sitename="Diffractor",
    authors="Keno Fischer and other contributors",
    pages=[
        "Introduction" => "index.md",
        "Reading List" => "reading_list.md",
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(
    repo = "github.com/JuliaDiff/Diffractor.jl.git",
    push_preview=true,
)
