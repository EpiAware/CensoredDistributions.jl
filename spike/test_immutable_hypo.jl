using ForwardDiff, Enzyme
using Enzyme: Const, Duplicated, autodiff, Reverse, set_runtime_activity
struct LeafV{T}
    ;
    mu::T;
    sig::T;
    nodes::Vector{Float64};
end
struct LeafT{T}
    ;
    mu::T;
    sig::T;
    nodes::NTuple{4, Float64};
end
score(l, x) = -0.5*((x-l.mu)/l.sig)^2 - log(l.sig)
function treeV(θ)
    (LeafV(θ[1], θ[2], [1.0, 2.0, 3.0, 4.0]), LeafV(θ[3], θ[4], [1.0, 2.0, 3.0, 4.0]))
end
function treeT(θ)
    (LeafT(θ[1], θ[2], (1.0, 2.0, 3.0, 4.0)), LeafT(θ[3], θ[4], (1.0, 2.0, 3.0, 4.0)))
end
fV(θ, x) = score(treeV(θ)[1], x[1])+score(treeV(θ)[2], x[2])
fT(θ, x) = score(treeT(θ)[1], x[1])+score(treeT(θ)[2], x[2])
θ=[0.5, 1.0, 0.3, 2.0];
x=[1.0, 2.0]
for (nm, f) in (("Vector nodes", fV), ("Tuple nodes", fT))
    dθ=zero(θ);
    print(nm, ": ")
    try
        autodiff(set_runtime_activity(Reverse), Const(f), Duplicated(θ, dθ), Const(x))
        ref=ForwardDiff.gradient(t->f(t, x), θ)
        println(isapprox(dθ, ref; rtol = 1e-6) ? "OK" : "WRONG $dθ vs $ref")
    catch e
        ;
        println("ERR ", first(split(sprint(showerror, e), '\n')));
    end
end
