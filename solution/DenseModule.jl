module DenseModule
using Random, LinearAlgebra

export DenseLayer, init_dense_layer, backward_pass, relu, relu_grad, sigmoid, sigmoid_grad, identity, identity_grad

mutable struct DenseLayer
    weights::Array{Float32,2}
    biases::Array{Float32,1}
    grad_weights::Union{Nothing,Array{Float32,2}}
    grad_biases::Union{Nothing,Array{Float32,1}}
    activation::Function
    activation_grad::Function
    activations::Array{Float32,2}
    inputs::Array{Float32,2}
end

function relu(x)
    return max.(0, x)
end

function relu_grad(x)
    return float.(x .> 0)
end

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function sigmoid_grad(x)
    s = sigmoid(x)
    return s .* (1 .- s)
end

function identity(x)
    return x
end

function identity_grad(x)
    return ones(size(x))
end

# Updated to handle different array types
function (layer::DenseLayer)(input)
    # Convert input to a standard Array{Float32,2} if it's not already
    input_matrix = convert(Array{Float32,2}, input)
    
    # Store intermediate values needed for backpropagation
    layer.inputs = input_matrix  # Save input to use in the backward pass
    layer.activations = layer.activation(layer.weights * input_matrix .+ layer.biases)
    
    return layer.activations
end

function init_dense_layer(input_dim::Int, output_dim::Int, activation::Function, activation_grad::Function, seedy::Int)
    Random.seed!(seedy)

    # Xavier/Glorot initialization for better convergence
    scale = sqrt(2.0 / (input_dim + output_dim))
    weights = scale * randn(Float32, output_dim, input_dim)
    
    biases = zeros(Float32, output_dim)
    grad_weights = zeros(Float32, output_dim, input_dim)
    grad_biases = zeros(Float32, output_dim)
    activations = zeros(Float32, output_dim, 1)
    inputs = zeros(Float32, input_dim, 1)
    return DenseLayer(weights, biases, grad_weights, grad_biases, activation, activation_grad, activations, inputs)
end

# Implementing the backward pass for the dense layer
function backward_pass(layer::DenseLayer, d_output::Array{Float32,2})
    # Apply the derivative of the activation function
    d_activation = layer.activation_grad(layer.activations) .* d_output

    # Calculate gradients
    d_weights = d_activation * layer.inputs'
    d_biases = sum(d_activation, dims=2)
    d_input = layer.weights' * d_activation
    
    # Update gradients
    layer.grad_weights .+= d_weights
    layer.grad_biases .+= d_biases[:, 1]  # Convert to vector

    return Float32.(d_input)
end

end