using ADFixtures, ADTypes, DifferentiationInterface
using ForwardDiff, Enzyme
scens = ADFixtures.scenarios(with_reference = true)
s = only(filter(x->x.name=="Nested tree censored observed logpdf", scens))
ref = s.res1
tries = [
    ("rev + funcann=Duplicated",
        AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Duplicated)),
    ("fwd + funcann=Duplicated",
        AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward),
            function_annotation = Enzyme.Duplicated)),
    ("rev + funcann=Const",
        AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Const))
]
for (nm, be) in tries
    print(nm, ": ")
    try
        g = DifferentiationInterface.gradient(s.f, be, s.x, s.contexts...)
        println(isapprox(g, ref; rtol = 1e-4) ? "OK" : "WRONG")
    catch e
        ;
        println("ERR ", first(split(sprint(showerror, e), '\n')));
    end
end
