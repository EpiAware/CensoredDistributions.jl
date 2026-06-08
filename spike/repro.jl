using ADFixtures
using ADTypes, DifferentiationInterface
using ForwardDiff, ReverseDiff, Enzyme, Mooncake

const targets = ["Nested tree censored observed logpdf",
    "Nested Competing tree conditioned logpdf"]

scens = ADFixtures.scenarios(with_reference = true)
sel = filter(s -> s.name in targets, scens)

backends = [
    ("ForwardDiff", AutoForwardDiff()),
    ("ReverseDiff", AutoReverseDiff(compile = false)),
    ("Mooncake rev", AutoMooncake(config = nothing)),
    ("Mooncake fwd", AutoMooncakeForward()),
    ("Enzyme rev", AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
    ("Enzyme fwd", AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Forward)))
]

for scen in sel
    println("\n=== ", scen.name, " ===")
    ref = scen.res1
    println("ref grad = ", ref)
    for (bn, be) in backends
        print(rpad(bn, 14), ": ")
        try
            g = DifferentiationInterface.gradient(scen.f, be, scen.x, scen.contexts...)
            ok = g isa AbstractVector && all(isfinite, g) &&
                 isapprox(g, ref; rtol = 5e-2, atol = 1e-6)
            println(ok ? "OK" : "WRONG  g=$(g)")
        catch e
            msg = sprint(showerror, e)
            println("ERROR  ", first(split(msg, '\n')))
        end
    end
end
