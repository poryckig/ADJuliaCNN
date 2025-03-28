module NetworkHandlers

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

export forward_pass_master, backward_pass_master

# Ultraszybki forward pass
@inline function forward_pass_master(net, input)
    current_output = input
    
    # Przetwarzanie przez każdą warstwę
    for layer in net
        current_output = layer(current_output)
    end
    
    return current_output
end

# Specjalizowany backward pass z typową warstwą jako argumentem
@inline function dispatch_backward_pass(layer, grad)
    # Znajdź moduł, w którym zdefiniowana jest warstwa
    module_name = parentmodule(typeof(layer))
    
    # Wywołaj odpowiednią funkcję backward_pass
    return module_name.backward_pass(layer, grad)
end

# Zoptymalizowany backward pass master
@inline function backward_pass_master(net, grad_output)
    current_grad = grad_output
    
    # Iteruj przez warstwy od końca
    for layer in Iterators.reverse(net)
        current_grad = dispatch_backward_pass(layer, current_grad)
    end
    
    return current_grad
end

end