superscript(i) = map(repr(i)) do c
    c   ==  '-' ? '\u207b' :
    c   ==  '1' ? '\u00b9' :
    c   ==  '2' ? '\u00b2' :
    c   ==  '3' ? '\u00b3' :
    c   ==  '4' ? '\u2074' :
    c   ==  '5' ? '\u2075' :
    c   ==  '6' ? '\u2076' :
    c   ==  '7' ? '\u2077' :
    c   ==  '8' ? '\u2078' :
    c   ==  '9' ? '\u2079' :
    c   ==  '0' ? '\u2070' :
    error("Unexpected Character")
end

subscript(i) = map(repr(i)) do c
    c == '0' ? '₀' :
    c == '1' ? '₁' :
    c == '2' ? '₂' :
    c == '3' ? '₃' :
    c == '4' ? '₄' :
    c == '5' ? '₅' :
    c == '6' ? '₆' :
    c == '7' ? '₇' :
    c == '8' ? '₈' :
    c == '9' ? '₉' :
    error("Unexpected Character")
end
