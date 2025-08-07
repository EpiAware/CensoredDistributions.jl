@testitem "Docstring completeness validation" tags = [:quality] begin
    using CensoredDistributions
    using Distributions

    # Get all exported symbols from CensoredDistributions
    exported_symbols = names(CensoredDistributions)

    # Key distribution types that should have interface method documentation
    # Access types via module to avoid scope issues in test environment
    key_types = [
        CensoredDistributions.PrimaryCensored,
        CensoredDistributions.IntervalCensored,
        CensoredDistributions.Weighted,
        CensoredDistributions.ExponentiallyTilted
    ]

    # Interface methods that should be documented for distribution types
    interface_methods = [
        # Basic interface
        params, minimum, maximum, insupport,
        # PDF/PMF interface
        pdf, logpdf,
        # CDF interface
        cdf, logcdf, ccdf, logccdf,
        # Other standard methods
        quantile, rand, mean, var, std
    ]

    @testset "Exported functions have docstrings" begin
        missing_docs = String[]

        for symbol in exported_symbols
            # Skip internal symbols
            if startswith(string(symbol), "_") || symbol == :CensoredDistributions
                continue
            end

            try
                obj = getfield(CensoredDistributions, symbol)
                # Check if it's a callable (function or type)
                if isa(obj, Function) || isa(obj, Type)
                    doc_str = string(@doc obj)
                    # Check if docstring is meaningful (not just the default)
                    if isempty(strip(doc_str)) ||
                       occursin("No documentation found", doc_str) ||
                       doc_str == "```\n$symbol\n```"
                        push!(missing_docs, string(symbol))
                    end
                end
            catch e
                # If we can't access the symbol, note it but don't fail
                push!(missing_docs, string(symbol) * " (access error)")
            end
        end

        if !isempty(missing_docs)
            @warn "Missing docstrings for exported symbols: $(join(missing_docs, ", "))"
        end
        @test isempty(missing_docs)
    end

    @testset "Interface methods have documentation" begin
        missing_interface_docs = Tuple{String, String}[]  # (type_name, method_name)

        for dist_type in key_types
            type_name = string(dist_type)

            for method in interface_methods
                method_name = string(method)

                try
                    # Check if method exists for this type
                    methods_list = methods(method, (dist_type,))

                    if !isempty(methods_list)
                        # Find methods that are defined in CensoredDistributions module
                        # (indicating they are specifically implemented for our types)
                        relevant_methods = filter(
                            m -> m.module == CensoredDistributions ||
                                 occursin("CensoredDistributions", string(m.module)),
                            methods_list
                        )

                        if !isempty(relevant_methods)
                            # Method is specifically implemented, check documentation
                            relevant_method = first(relevant_methods)
                            doc_str = string(@doc relevant_method)

                            if isempty(strip(doc_str)) ||
                               occursin("No documentation found", doc_str)
                                push!(missing_interface_docs, (type_name, method_name))
                            end
                        end
                    end
                catch e
                    # Method might not exist or might not be accessible
                    # This is not necessarily an error for interface completeness
                    continue
                end
            end
        end

        if !isempty(missing_interface_docs)
            grouped = Dict{String, Vector{String}}()
            for (type_name, method_name) in missing_interface_docs
                if !haskey(grouped, type_name)
                    grouped[type_name] = String[]
                end
                push!(grouped[type_name], method_name)
            end

            for (type_name, methods) in grouped
                @warn "Missing interface method docs for $type_name: $(join(methods, ", "))"
            end
        end

        @test isempty(missing_interface_docs)
    end

    @testset "Constructor functions have docstrings" begin
        constructor_functions = [
            primary_censored,
            interval_censored,
            double_interval_censored,
            weight
        ]

        missing_constructor_docs = String[]

        for func in constructor_functions
            func_name = string(func)
            doc_str = string(@doc func)

            if isempty(strip(doc_str)) ||
               occursin("No documentation found", doc_str) ||
               doc_str == "```\n$func_name\n```"
                push!(missing_constructor_docs, func_name)
            end
        end

        if !isempty(missing_constructor_docs)
            @warn "Missing docstrings for constructor functions: $(join(missing_constructor_docs, ", "))"
        end
        @test isempty(missing_constructor_docs)
    end

    @testset "Key utility functions have docstrings" begin
        utility_functions = [
            get_dist,
            get_dist_recursive,
            primarycensored_cdf,
            primarycensored_logcdf
        ]

        missing_utility_docs = String[]

        for func in utility_functions
            func_name = string(func)
            doc_str = string(@doc func)

            if isempty(strip(doc_str)) ||
               occursin("No documentation found", doc_str) ||
               doc_str == "```\n$func_name\n```"
                push!(missing_utility_docs, func_name)
            end
        end

        if !isempty(missing_utility_docs)
            @warn "Missing docstrings for utility functions: $(join(missing_utility_docs, ", "))"
        end
        @test isempty(missing_utility_docs)
    end
end
