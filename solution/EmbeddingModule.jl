module EmbeddingModule
using Random

export EmbeddingLayer, init_embedding_layer, backward_pass

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

mutable struct EmbeddingLayer
    weights::Array{Float32, 2}
    biases::Array{Float32, 1}
    grad_weights::Array{Float32, 2}
    grad_biases::Array{Float32, 1}
    vocab_size::Int
    last_indices::Union{Nothing, Array{Int, 1}}
    output_buffer::Union{Nothing, Array{Float32, 2}}  # Bufor wyjściowy
end

function init_embedding_layer(vocab_size::Int, embedding_dim::Int, seedy::Int)
    Random.seed!(seedy)
    
    # Inicjalizacja wag
    weights = 0.1f0 * randn(Float32, vocab_size, embedding_dim)
    biases = zeros(Float32, embedding_dim)
    
    grad_weights = zeros(Float32, vocab_size, embedding_dim)
    grad_biases = zeros(Float32, embedding_dim)
    
    return EmbeddingLayer(weights, biases, grad_weights, grad_biases, vocab_size, nothing, nothing)
end

# Szybka funkcja do bezpiecznego przetwarzania indeksów
@inline function safe_indices(indices, vocab_size)
    result = similar(indices)
    @inbounds for i in eachindex(indices)
        idx = indices[i]
        result[i] = (idx < 1 || idx > vocab_size) ? 2 : idx
    end
    return result
end

# Zoptymalizowany operator dla wektorów całkowitych
function (layer::EmbeddingLayer)(input::Array{Int, 1})
    # Przetwarzanie indeksów z bezpieczną obsługą
    safe_input = safe_indices(input, layer.vocab_size)
    
    # Przechowaj indeksy do backward pass
    layer.last_indices = safe_input
    
    # Przygotuj bufor wyjściowy
    seq_length = length(safe_input)
    embedding_dim = size(layer.weights, 2)
    
    if isnothing(layer.output_buffer) || size(layer.output_buffer) != (seq_length, embedding_dim)
        layer.output_buffer = zeros(Float32, seq_length, embedding_dim)
    else
        fill!(layer.output_buffer, 0.0f0)
    end
    
    # Szybkie wyszukiwanie embeddingów
    @inbounds for i in 1:seq_length
        idx = safe_input[i]
        @inbounds for j in 1:embedding_dim
            layer.output_buffer[i, j] = layer.weights[idx, j] + layer.biases[j]
        end
    end
    
    return layer.output_buffer
end

# Obsługa wektorów Float32
function (layer::EmbeddingLayer)(input::Array{Float32, 1})
    # Konwersja na Int
    input_int = round.(Int, input)
    return layer(input_int)
end

# Zoptymalizowany backward pass
function backward_pass(layer::EmbeddingLayer, grad_output::Array{<:AbstractFloat, 2})
    if isnothing(layer.last_indices)
        error("No input indices stored. Forward pass must be called before backward pass.")
    end
    
    # Gradient wejściowy (zwykle nieużywany dla warstwy embeddingowej)
    grad_input = zeros(Float32, length(layer.last_indices))
    
    # Aktualizacja gradientów wag - zoptymalizowana wersja
    @inbounds for (pos, idx) in enumerate(layer.last_indices)
        # Pomijamy nieprawidłowe indeksy
        if 1 <= idx <= layer.vocab_size
            @inbounds for j in 1:size(grad_output, 2)
                layer.grad_weights[idx, j] += grad_output[pos, j]
            end
        end
    end
    
    # Aktualizacja gradientów biasów - szybka wersja
    @inbounds for j in 1:length(layer.grad_biases)
        sum_j = 0.0f0
        @inbounds for i in 1:size(grad_output, 1)
            sum_j += grad_output[i, j]
        end
        layer.grad_biases[j] += sum_j
    end
    
    return grad_input
end

# Obsługa różnych typów gradientów
function backward_pass(layer::EmbeddingLayer, grad_output)
    # Zapewnij właściwy kształt do przetwarzania
    if ndims(grad_output) == 1
        grad_reshaped = reshape(Float32.(grad_output), :, 1)
    else
        grad_reshaped = Float32.(grad_output)
    end
    
    return backward_pass(layer, grad_reshaped)
end

end