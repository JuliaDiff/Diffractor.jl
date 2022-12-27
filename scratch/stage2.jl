using Cthulhu
using Diffractor
using Diffractor: ADInterpreter
using Diffractor: var"'", ∂⃖

function foo(x)
    sin(x)
end
bar(x) = foo(x)

diffsin(x) = bar'(x)

diffsin(1.0)

function do_the_thing()
    interp = ADInterpreter()
    mi = Cthulhu.get_specialization(Tuple{map(Core.Typeof, (diffsin, 2.0))...})
    Cthulhu.do_typeinf!(interp, mi)
    (interp, mi)
end
(interp, mi) = do_the_thing();
Diffractor.codegen(interp, Diffractor.ADCursor(0, mi))(1.0)
