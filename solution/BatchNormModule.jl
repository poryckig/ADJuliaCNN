module BatchNormModule

using Statistics

export BatchNormLayer, init_batchnorm_layer, backward_pass

# Struktura warstwy normalizacji wsadowej
mutable struct BatchNormLayer
    gamma::Array{Float32, 1}        # Parametry skalujące
    beta::Array{Float32, 1}         # Parametry przesunięcia
    epsilon::Float32                # Mała wartość dla stabilności numerycznej
    momentum::Float32               # Współczynnik momentu dla średnich ruchomych
    running_mean::Array{Float32, 1} # Średnia bieżąca do fazy testowej
    running_var::Array{Float32, 1}  # Wariancja bieżąca do fazy testowej
    grad_gamma::Array{Float32, 1}   # Gradient parametrów gamma
    grad_beta::Array{Float32, 1}    # Gradient parametrów beta
    cache::Dict{Symbol, Any}        # Pamięć podręczna dla propagacji wstecznej
    training::Bool                  # Tryb: uczenie czy testowanie
end

# Inicjalizacja warstwy normalizacji wsadowej
function init_batchnorm_layer(num_features::Int)
    gamma = ones(Float32, num_features)
    beta = zeros(Float32, num_features)
    epsilon = 1.0f-5
    momentum = 0.1f0
    running_mean = zeros(Float32, num_features)
    running_var = ones(Float32, num_features)
    grad_gamma = zeros(Float32, num_features)
    grad_beta = zeros(Float32, num_features)
    cache = Dict{Symbol, Any}()
    training = true
    
    return BatchNormLayer(gamma, beta, epsilon, momentum, running_mean, 
                         running_var, grad_gamma, grad_beta, cache, training)
end

# Przełączanie trybu uczenia/testowania
function set_training_mode!(layer::BatchNormLayer, training::Bool)
    layer.training = training
end

# Operator wywołania warstwy (forward pass)
function (layer::BatchNormLayer)(input::Array{Float32, 2})
    seq_length, num_features = size(input)
    
    if layer.training
        # Obliczanie statystyk batchowych
        batch_mean = vec(mean(input, dims=1))
        batch_var = vec(var(input, dims=1, corrected=false))
        
        # Aktualizacja średnich ruchomych
        layer.running_mean .= (1 - layer.momentum) .* layer.running_mean .+ layer.momentum .* batch_mean
        layer.running_var .= (1 - layer.momentum) .* layer.running_var .+ layer.momentum .* batch_var
        
        # Normalizacja
        x_centered = input .- reshape(batch_mean, 1, :)
        x_scaled = x_centered ./ sqrt.(reshape(batch_var .+ layer.epsilon, 1, :))
        
        # Skalowanie i przesunięcie
        out = reshape(layer.gamma, 1, :) .* x_scaled .+ reshape(layer.beta, 1, :)
        
        # Zapisywanie danych dla propagacji wstecznej
        layer.cache[:x_scaled] = x_scaled
        layer.cache[:batch_mean] = batch_mean
        layer.cache[:batch_var] = batch_var
        layer.cache[:input] = input
        
        return out
    else
        # Normalizacja ze średnimi ruchomymi (tryb testowy)
        x_normalized = (input .- reshape(layer.running_mean, 1, :)) ./ 
                       sqrt.(reshape(layer.running_var .+ layer.epsilon, 1, :))
        
        return reshape(layer.gamma, 1, :) .* x_normalized .+ reshape(layer.beta, 1, :)
    end
end

# Propagacja wsteczna
function backward_pass(layer::BatchNormLayer, grad_output::AbstractArray{<:AbstractFloat})
    # Upewnij się, że grad_output jest macierzą 2D
    if ndims(grad_output) == 1
        grad_output = reshape(grad_output, 1, :)  # Przekształć wektor w macierz 1xN
    end
    
    if !layer.training || !haskey(layer.cache, :x_scaled)
        error("BatchNorm backward pass wymaga trybu training=true oraz danych w cache")
    end
    
    # Pobierz zapisane dane
    x_scaled = layer.cache[:x_scaled]
    batch_mean = layer.cache[:batch_mean]
    batch_var = layer.cache[:batch_var]
    input = layer.cache[:input]
    
    # Dopasuj kształt gradientu wyjściowego do kształtu wejścia
    if size(grad_output, 1) != size(input, 1)
        # Jeśli gradient ma inny kształt niż wejście, rozszerzamy go
        grad_output = repeat(grad_output, outer=(size(input, 1), 1))
    end
    
    # Kształt danych
    m, D = size(input)
    
    # Gradienty względem parametrów gamma i beta
    dgamma = vec(sum(grad_output .* x_scaled, dims=1))
    dbeta = vec(sum(grad_output, dims=1))
    
    # Akumuluj gradienty
    layer.grad_gamma .+= dgamma
    layer.grad_beta .+= dbeta
    
    # Gradient względem x_scaled
    dx_scaled = grad_output .* reshape(layer.gamma, 1, :)
    
    # Gradient względem wariancji
    dsigma2 = sum(dx_scaled .* (input .- reshape(batch_mean, 1, :)) .* 
              reshape(-0.5f0 .* (batch_var .+ layer.epsilon).^(-1.5f0), 1, :), dims=1)
    
    # Gradient względem średniej
    dmu = sum(dx_scaled .* reshape(-1.0f0 ./ sqrt.(batch_var .+ layer.epsilon), 1, :), dims=1) + 
          dsigma2 .* reshape(-2.0f0 .* sum(input .- reshape(batch_mean, 1, :), dims=1) ./ m, 1, :)
    
    # Gradient względem wejścia
    dx = dx_scaled .* reshape(1.0f0 ./ sqrt.(batch_var .+ layer.epsilon), 1, :) + 
         reshape(dsigma2, 1, :) .* 2.0f0 .* (input .- reshape(batch_mean, 1, :)) ./ m + 
         reshape(dmu, 1, :) ./ m
    
    return dx
end

end