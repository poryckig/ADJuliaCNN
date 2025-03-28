module DenseModule
using Random, LinearAlgebra

export DenseLayer, init_dense_layer, backward_pass, relu, relu_grad, sigmoid, sigmoid_grad, identity, identity_grad

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

mutable struct DenseLayer
    weights::Array{Float32,2}
    biases::Array{Float32,1}
    grad_weights::Array{Float32,2}
    grad_biases::Array{Float32,1}
    activation::Function
    activation_grad::Function
    pre_activations::Array{Float32,2}
    activations::Array{Float32,2}
    inputs::Array{Float32,2}
    # Bufory dla wydajności
    output_buffer::Union{Nothing,Array{Float32,2}}
end

# Funkcje aktywacji - specjalnie zoptymalizowane
@inline function relu(x::Float32)
    return max(0.0f0, x)
end

@inline function relu(x::Array{Float32})
    return max.(0.0f0, x)
end

@inline function relu_grad(x::Float32)
    return Float32(x > 0.0f0)
end

@inline function relu_grad(x::Array{Float32})
    return Float32.(x .> 0.0f0)
end

@inline function sigmoid(x::Float32)
    if x >= 0
        t = exp(-x)
        return 1.0f0 / (1.0f0 + t)
    else
        t = exp(x)
        return t / (1.0f0 + t)
    end
end

@inline function sigmoid(x::Array{Float32})
    result = similar(x)
    @inbounds for i in eachindex(x)
        result[i] = sigmoid(x[i])
    end
    return result
end

@inline function sigmoid_grad(x::Float32)
    s = sigmoid(x)
    return s * (1.0f0 - s)
end

@inline function sigmoid_grad(x::Array{Float32})
    result = similar(x)
    @inbounds for i in eachindex(x)
        result[i] = sigmoid_grad(x[i])
    end
    return result
end

@inline function identity(x)
    return x
end

@inline function identity_grad(x)
    return ones(Float32, size(x))
end

# Ultraszybka konwersja do macierzy
@inline function fast_matrix_convert(input)
    if isa(input, Array{Float32,2})
        return input
    elseif isa(input, Vector{Float32})
        return reshape(input, :, 1)
    else
        # Szybka konwersja
        arr = convert(Array{Float32}, input)
        return ndims(arr) == 1 ? reshape(arr, :, 1) : arr
    end
end

# Szybkie mnożenie macierzy z BLAS
@inline function matmul_optimized!(output, weights, input, biases)
    # Użyj BLAS gemm! dla najlepszej wydajności
    mul!(output, weights, input)
    
    # Dodaj bias (wektoryzowane)
    output_rows, output_cols = size(output)
    @inbounds for j in 1:output_cols
        @simd for i in 1:output_rows
            output[i, j] += biases[i]
        end
    end
    
    return output
end

# Optymalizowana aplikacja funkcji aktywacji
@inline function apply_activation!(output, activation_fn, input)
    @inbounds for i in eachindex(input)
        output[i] = activation_fn(input[i])
    end
    return output
end

# Ultraszybki operator wywołania
function (layer::DenseLayer)(input)
    # Szybka konwersja do odpowiedniego formatu macierzy
    input_matrix = fast_matrix_convert(input)
    
    # Zachowaj wejście dla backward pass
    layer.inputs = input_matrix
    
    # Oblicz wymiary wyjścia
    output_rows = size(layer.weights, 1)
    output_cols = size(input_matrix, 2)
    
    # Przygotuj lub użyj buforów
    if isnothing(layer.output_buffer) || size(layer.output_buffer) != (output_rows, output_cols)
        layer.output_buffer = zeros(Float32, output_rows, output_cols)
        layer.pre_activations = zeros(Float32, output_rows, output_cols)
        layer.activations = zeros(Float32, output_rows, output_cols)
    end
    
    # Wykonaj mnożenie macierzy i dodaj bias
    matmul_optimized!(layer.pre_activations, layer.weights, input_matrix, layer.biases)
    
    # Zastosuj funkcję aktywacji
    apply_activation!(layer.activations, layer.activation, layer.pre_activations)
    
    return layer.activations
end

# Inicjalizacja warstwy gęstej
function init_dense_layer(input_dim::Int, output_dim::Int, activation::Function, activation_grad::Function, seedy::Int)
    Random.seed!(seedy)

    # Xavier inicjalizacja
    scale = sqrt(2.0f0 / (input_dim + output_dim))
    weights = scale * randn(Float32, output_dim, input_dim)
    
    biases = zeros(Float32, output_dim)
    grad_weights = zeros(Float32, output_dim, input_dim)
    grad_biases = zeros(Float32, output_dim)
    pre_activations = zeros(Float32, output_dim, 1)
    activations = zeros(Float32, output_dim, 1)
    inputs = zeros(Float32, input_dim, 1)
    
    return DenseLayer(weights, biases, grad_weights, grad_biases, 
                     activation, activation_grad, pre_activations, 
                     activations, inputs, nothing)
end

# Zoptymalizowany backward pass
function backward_pass(layer::DenseLayer, d_output)
    # Upewnij się, że d_output ma odpowiedni kształt
    if ndims(d_output) == 1
        d_output = reshape(d_output, :, 1)
    end
    
    # Zastosuj gradient funkcji aktywacji
    d_activation = similar(d_output)
    @inbounds for i in eachindex(d_output)
        d_activation[i] = layer.activation_grad(layer.pre_activations[i]) * d_output[i]
    end
    
    # Oblicz gradienty - szybkie mnożenie macierzy z BLAS
    mul!(layer.grad_weights, d_activation, layer.inputs', 1.0f0, 1.0f0)  # Akumuluj
    
    # Oblicz gradienty biasów
    @inbounds for j in 1:size(d_activation, 2)
        @inbounds for i in 1:size(d_activation, 1)
            layer.grad_biases[i] += d_activation[i, j]
        end
    end
    
    # Oblicz gradienty wejścia
    d_input = layer.weights' * d_activation
    
    return d_input
end

end