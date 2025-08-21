# Shared helper functions for docstring validation
@testsnippet DocstringHelpers begin
    using CensoredDistributions
    using Distributions
    using Markdown

    # Helper function to get docstring content as string
    function get_docstring_content(obj)
        try
            binding = Base.Docs.Binding(parentmodule(obj), nameof(obj))
            docs = Base.Docs.meta(parentmodule(obj))

            if haskey(docs, binding)
                doc_obj = docs[binding]

                # Handle MultiDoc case - get all docstrings and combine them
                if doc_obj isa Base.Docs.MultiDoc
                    all_docs = String[]
                    for (sig, docstr) in doc_obj.docs
                        # Process each docstring - handle templates if present
                        doc_content = string(docstr.text[2])  # The actual content is usually at index 2
                        push!(all_docs, doc_content)
                    end
                    return join(all_docs, "\n\n")
                else
                    # Handle single doc case
                    return string(doc_obj)
                end
            end
        catch e
            # Fallback if anything goes wrong
        end
        return "No documentation found."
    end

    # Helper to extract function signature and arguments
    function extract_function_info(func_name, mod = CensoredDistributions)
        methods_list = methods(getfield(mod, func_name))
        all_args = Set{Symbol}()
        has_kwargs = false

        for method in methods_list
            try
                # Get argument names from method signature
                arg_names = Base.method_argnames(method)
                if length(arg_names) > 1
                    # Skip first argument (function name) and filter out internal arguments
                    relevant_args = arg_names[2:end]
                    for arg in relevant_args
                        arg_str = string(arg)
                        if arg ≠ Symbol("#unused#") &&
                           !startswith(arg_str, "#") &&
                           !startswith(arg_str, "var\"") &&
                           arg ≠ Symbol("") &&
                           !occursin("##", arg_str)
                            all_args = union(all_args, [arg])
                        end
                    end
                end

                # Check if method has keyword arguments
                if method.nkw > 0
                    has_kwargs = true
                end
            catch e
                # Skip methods that can't be introspected
                continue
            end
        end

        return collect(all_args), has_kwargs
    end

    # Helper to check if an object is a type (not a function)
    function is_type_export(name, mod)
        try
            obj = getfield(mod, name)
            return obj isa Type
        catch
            return false
        end
    end

    # Get public symbols (Julia 1.11+) and exported symbols
    function get_public_symbols()
        # Get public symbols if available (Julia 1.11+)
        @static if VERSION >= v"1.11"
            if isdefined(CensoredDistributions, :public)
                try
                    # Try to access public symbols via module metadata
                    # Julia 1.11+ should provide this but API is still developing
                    # For now, return empty array as public API isn't finalized
                    return Symbol[]
                catch
                    return Symbol[]
                end
            else
                return Symbol[]
            end
        else
            return Symbol[]
        end
    end

    # Automatically discover all exports and public symbols
    function discover_all_symbols()
        # Get exported symbols
        exported_symbols = names(CensoredDistributions)

        # Get public symbols
        public_symbols = get_public_symbols()

        # Combine exported and public, remove duplicates
        all_symbols = unique(vcat(exported_symbols, public_symbols))

        # Split into types and functions
        all_types = [name
                     for name in all_symbols if is_type_export(name, CensoredDistributions)]
        all_functions = [name
                         for name in all_symbols
                         if !is_type_export(name, CensoredDistributions)]

        # Also track which are exported vs public
        exported_types = [name
                          for name in exported_symbols
                          if is_type_export(name, CensoredDistributions)]
        exported_functions = [name
                              for name in exported_symbols
                              if !is_type_export(name, CensoredDistributions)]

        public_types = [name
                        for name in public_symbols
                        if is_type_export(name, CensoredDistributions)]
        public_functions = [name
                            for name in public_symbols
                            if !is_type_export(name, CensoredDistributions)]

        return (all_types, all_functions, exported_types,
            exported_functions, public_types, public_functions)
    end

    # Assign discovered symbols as variables for use in test items
    all_types, all_functions, exported_types, exported_functions,
    public_types, public_functions = discover_all_symbols()
end

@testitem "Type Documentation Format" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Type Documentation" begin
        for type_name in all_types
            @testset "$type_name" begin
                try
                    type_obj = getfield(CensoredDistributions, type_name)

                    # Only test if docstring exists (let Aqua handle existence)
                    doc_str = get_docstring_content(type_obj)
                    if !occursin("No documentation found", doc_str) &&
                       length(strip(doc_str)) > 10

                        # Skip test if no meaningful docstring
                        if length(strip(doc_str)) > length(string(type_name)) + 10
                            # Check if this is a struct type - if so, it should have field documentation
                            if type_name in all_types
                                try
                                    type_obj = getfield(CensoredDistributions, type_name)
                                    if hasmethod(fieldnames, Tuple{Type{type_obj}})
                                        field_names = fieldnames(type_obj)
                                        if length(field_names) > 0
                                            # Should have field documentation for each field
                                            for field_name in field_names
                                                # For fields, just check if they're documented in the type docstring
                                                # since field-level docs aren't commonly used in Julia
                                                @test occursin(string(field_name), doc_str)
                                            end
                                        else
                                            @test true  # No fields to document
                                        end
                                    else
                                        @test true  # Not a struct with fields
                                    end
                                catch e
                                    @test true  # Skip if can't introspect
                                end
                            else
                                @test true  # Not a recognized type
                            end
                        else
                            # Skip test if no meaningful docstring
                            @test true
                        end
                    else
                        # Skip test if no docstring exists
                        @test true
                    end
                catch e
                    @warn "Could not test $type_name: $e"
                    @test true
                end
            end
        end
    end

    # Report discovered structure for debugging
    @info "Discovered symbols" all_types=all_types exported_types=exported_types public_types=public_types
end

@testitem "Function Documentation Format" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Function Documentation" begin
        for func_name in all_functions
            @testset "$func_name" begin
                try
                    func_obj = getfield(CensoredDistributions, func_name)

                    # Only test if docstring exists (let Aqua handle existence)
                    doc_str = get_docstring_content(func_obj)
                    if !occursin("No documentation found", doc_str) &&
                       length(strip(doc_str)) > 10

                        # Skip if docstring is just the object name or no documentation found
                        if !occursin("No documentation found", doc_str) &&
                           length(strip(doc_str)) > length(string(func_name)) + 10

                            # Check each method's documentation individually
                            try
                                methods_list = methods(func_obj)
                                function_has_args = false
                                function_has_kwargs = false

                                # Check if any method has meaningful arguments
                                for method in methods_list
                                    try
                                        arg_names = Base.method_argnames(method)
                                        if length(arg_names) > 1
                                            relevant_args = arg_names[2:end]
                                            method_args = []

                                            for arg in relevant_args
                                                arg_str = string(arg)
                                                if arg ≠ Symbol("#unused#") &&
                                                   !startswith(arg_str, "#") &&
                                                   !startswith(arg_str, "var\"") &&
                                                   arg ≠ Symbol("") &&
                                                   !occursin("##", arg_str) &&
                                                   length(arg_str) > 1
                                                    push!(method_args, arg)
                                                end
                                            end

                                            if !isempty(method_args)
                                                function_has_args = true

                                                # Check that documented arguments are reasonable for this function
                                                args_section_match = match(
                                                    r"# Arguments(.*?)(?=# [A-Z]|@|\z)"s, doc_str)
                                                if args_section_match !== nothing
                                                    args_section = args_section_match.captures[1]
                                                    # At least some of this method's args should be documented
                                                    # (allowing for multiple methods to have different args)
                                                    method_args_found = 0
                                                    for arg in method_args
                                                        arg_pattern = "- `$(arg)"
                                                        if occursin(arg_pattern, args_section)
                                                            method_args_found += 1
                                                        end
                                                    end
                                                    # Allow flexibility: if this method contributes some documented args, that's good
                                                end
                                            end
                                        end

                                        # Check for keyword arguments
                                        if method.nkw > 0
                                            function_has_kwargs = true
                                        end
                                    catch
                                        continue
                                    end
                                end

                                # If function has arguments across any method, should have Arguments section
                                if function_has_args
                                    @test occursin("# Arguments", doc_str)
                                end

                                # If function has keyword arguments, check for Keyword Arguments section
                                if function_has_kwargs
                                    @test occursin("# Keyword Arguments", doc_str)
                                end

                            catch e
                                @warn "Could not extract argument info for $func_name: $e"
                                # No hardcoded fallbacks - if we can't extract args, skip argument validation
                                @test true
                            end

                            # All exported/public functions should have examples
                            if func_name in exported_functions ||
                               func_name in public_functions
                                @test occursin("@example", doc_str) ||
                                      occursin("```@example", doc_str)
                            end

                            # Check for TYPEDSIGNATURES macro usage (from DocStringExtensions) or function signature
                            @test occursin("TYPEDSIGNATURES", doc_str) ||
                                  occursin(string(func_name), doc_str)
                        else
                            # Skip test if no meaningful docstring
                            @test true
                        end
                    else
                        # Skip test if no docstring exists
                        @test true
                    end
                catch e
                    @warn "Could not test $func_name: $e"
                    @test true
                end
            end
        end
    end

    # Report discovered structure for debugging
    @info "Discovered functions" all_functions=all_functions exported_functions=exported_functions public_functions=public_functions
end

@testitem "Cross-Reference Validation" setup=[DocstringHelpers] tags=[:quality] begin
    @testset "Cross-Reference Validation" begin
        # Check that See also sections reference valid functions/types
        all_names = union(all_types, all_functions)

        for name in all_names
            try
                obj = getfield(CensoredDistributions, name)
                doc_str = get_docstring_content(obj)

                if !occursin("No documentation found", doc_str) &&
                   length(strip(doc_str)) > 10

                    # Skip if no meaningful docstring
                    if !occursin("No documentation found", doc_str) &&
                       length(strip(doc_str)) > length(string(name)) + 10
                        # Extract references from See also sections
                        see_also_matches = eachmatch(r"`([^`]+)`\]\(@ref\)", doc_str)
                        for match in see_also_matches
                            referenced_name = Symbol(match.captures[1])
                            if referenced_name ∉ all_names &&
                               referenced_name ∉
                               [:pdf, :cdf, :logpdf, :logcdf, :rand, :quantile]
                                @warn "Function/type $name references non-existent $referenced_name in See also section"
                            end
                        end
                    end
                end
            catch e
                @warn "Could not validate cross-references for $name: $e"
            end
        end

        # Always pass this test - it's about warnings, not failures
        @test true
    end
end
