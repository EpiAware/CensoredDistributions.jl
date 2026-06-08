using ADFixtures, ADTypes, DifferentiationInterface
using ForwardDiff, Enzyme
scens = ADFixtures.scenarios(with_reference = true)
s = only(filter(x -> x.name == "Nested tree censored observed logpdf", scens))
ref = s.res1
# DI style with Constant context (as the test does)
for (nm,
    be) in (("Enz rev", AutoEnzyme(mode = set_runtime_activity(Reverse))),
    ("Enz fwd", AutoEnzyme(mode = set_runtime_activity(Forward))))
    print(nm, ": ")
    try
        g = DifferentiationInterface.gradient(s.f, be, s.x, s.contexts...)
        println(isapprox(g, ref; rtol = 1e-4) ? "OK" : "WRONG")
    catch e
        println("ERR ", first(split(sprint(showerror, e), '\n')))
    end
end
# Raw Enzyme without DI, ev captured
ev = s.contexts[1].data
f(θ) = s.f(θ, ev)
print("raw Enz rev (captured ev): ")
try
    g = Enzyme.gradient(set_runtime_activity(Reverse), f, s.x)[1]
    println(isapprox(g, ref; rtol = 1e-4) ? "OK" : "WRONG")
catch e
    ;
    println("ERR ", first(split(sprint(showerror, e), '\n')));
end
