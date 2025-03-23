module EmbeddingModule
using Random

export EmbeddingLayer, init_embedding_layer, backward_pass

mutable struct EmbeddingLayer
    weights::Array{Float32, 2}
    biases::Array{Float32, 1}
    grad_weights::Array{Float32, 2}
    grad_biases::Array{Float32, 1}
    vocab_size::Int  # Store vocabulary size
    last_indices::Union{Nothing, Array{Int, 1}}
end

function init_embedding_layer(vocab_size::Int, embedding_dim::Int, seedy::Int)
    Random.seed!(seedy)
    
    weights = 0.1 * randn(Float32, vocab_size, embedding_dim)
    biases = zeros(Float32, embedding_dim)
    
    grad_weights = zeros(Float32, vocab_size, embedding_dim)
    grad_biases = zeros(Float32, embedding_dim)
    
    return EmbeddingLayer(weights, biases, grad_weights, grad_biases, vocab_size, nothing)
end

# Ensure indices are within valid range
function safe_indices(indices, vocab_size)
    # Replace out-of-bounds indices with 2 (the <UNK> token)
    return [idx < 1 || idx > vocab_size ? 2 : idx for idx in indices]
end

# Handle vectors of integers
function (layer::EmbeddingLayer)(input::Array{Int, 1})
    # Clean indices to ensure they're within bounds
    safe_input = safe_indices(input, layer.vocab_size)
    
    # Store safe indices for backward pass
    layer.last_indices = copy(safe_input)
    
    # For each word index, look up its embedding vector
    embeddings = layer.weights[safe_input, :]
    
    # Add biases (broadcasting across sequence)
    embeddings .+= layer.biases'
    
    return Float32.(embeddings)
end

# Handle vectors of Float32
function (layer::EmbeddingLayer)(input::Array{Float32, 1})
    # Convert to Int and handle safely
    input_int = round.(Int, input)
    return layer(input_int)
end

# Handle matrices of integers (batch processing)
function (layer::EmbeddingLayer)(input::Array{Int, 2})
    batch_size, sequence_length = size(input)
    embedding_dim = size(layer.weights, 2)
    
    # Clean and reshape
    flat_input = vec(input)
    safe_flat_input = safe_indices(flat_input, layer.vocab_size)
    
    # Store for backward pass
    layer.last_indices = copy(safe_flat_input)
    
    # Look up embeddings
    embeddings = layer.weights[safe_flat_input, :]
    
    # Reshape to batch_size × sequence_length × embedding_dim
    embeddings = reshape(embeddings, batch_size, sequence_length, embedding_dim)
    
    # Add biases
    for i in 1:batch_size
        for j in 1:sequence_length
            embeddings[i, j, :] .+= layer.biases
        end
    end
    
    return Float32.(embeddings)
end

# Handle matrices of Float32
function (layer::EmbeddingLayer)(input::Array{Float32, 2})
    input_int = round.(Int, input)
    return layer(input_int)
end

function backward_pass(layer::EmbeddingLayer, grad_output::Array{Float32, 2})
    # grad_output shape: sequence_length × embedding_dim
    
    if isnothing(layer.last_indices)
        error("No input indices stored. Forward pass must be called before backward pass.")
    end
    
    grad_input = zeros(Float32, length(layer.last_indices))
    
    for (pos, idx) in enumerate(layer.last_indices)
        # Skip invalid indices (should be already handled but just in case)
        if 1 <= idx <= layer.vocab_size
            layer.grad_weights[idx, :] .+= grad_output[pos, :]
        end
    end
    
    layer.grad_biases .+= sum(grad_output, dims=1)[:]
    
    return grad_input
end

function backward_pass(layer::EmbeddingLayer, grad_output::Array{Float32, 3})
    # For batch processing
    batch_size, sequence_length, embedding_dim = size(grad_output)
    
    if isnothing(layer.last_indices)
        error("No input indices stored. Forward pass must be called before backward pass.")
    end
    
    flat_grad_output = reshape(grad_output, :, embedding_dim)
    
    grad_input = zeros(Float32, length(layer.last_indices))
    
    for (pos, idx) in enumerate(layer.last_indices)
        # Skip invalid indices
        if 1 <= idx <= layer.vocab_size
            layer.grad_weights[idx, :] .+= flat_grad_output[pos, :]
        end
    end
    
    layer.grad_biases .+= sum(flat_grad_output, dims=1)[:]
    
    return grad_input
end

end