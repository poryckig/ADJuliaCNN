module PoolModule

export MaxPoolLayer, init_pool_layer, backward_pass

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

mutable struct MaxPoolLayer
    pool_length::Int
    stride::Int
    max_indices::Union{Nothing,Array{Int,2}}
    input_size::Union{Nothing, Tuple{Int, Int}}
    output_buffer::Union{Nothing,Array{Float32,2}}  # Bufor wyjściowy
end

function init_pool_layer(pool_length::Int, stride::Int)
    MaxPoolLayer(pool_length, stride, nothing, nothing, nothing)
end

# Ultraszybki pooling
@inline function fast_maxpool!(output, max_indices, input, pool_length, stride)
    input_length, num_channels = size(input)
    output_length = div(input_length - pool_length, stride) + 1
    
    # Inicjalizuj wyjście
    fill!(output, -Inf32)
    
    # Szybki pooling z wektoryzacją
    @inbounds for c in 1:num_channels
        @inbounds for pos in 1:stride:input_length-pool_length+1
            output_idx = div(pos - 1, stride) + 1
            
            # Znajdź maksimum w oknie
            max_val = -Inf32
            max_pos = 0
            
            @inbounds for j in 0:pool_length-1
                val = input[pos+j, c]
                if val > max_val
                    max_val = val
                    max_pos = pos+j
                end
            end
            
            output[output_idx, c] = max_val
            max_indices[output_idx, c] = max_pos
        end
    end
    
    return output, max_indices
end

# Zoptymalizowany operator wywołania
function (layer::MaxPoolLayer)(input::Array{Float32,2})
    # Przechowaj wymiar wejścia
    layer.input_size = size(input)
    input_length, num_channels = layer.input_size
    
    # Oblicz wymiar wyjścia
    output_length = div(input_length - layer.pool_length, layer.stride) + 1
    
    # Przygotuj bufory
    if isnothing(layer.output_buffer) || size(layer.output_buffer) != (output_length, num_channels)
        layer.output_buffer = zeros(Float32, output_length, num_channels)
        layer.max_indices = Array{Int,2}(undef, output_length, num_channels)
    end
    
    # Wykonaj pooling
    fast_maxpool!(layer.output_buffer, layer.max_indices, input, layer.pool_length, layer.stride)
    
    return layer.output_buffer
end

# Zoptymalizowany backward pass
function backward_pass(layer::MaxPoolLayer, grad_output::Array{Float32,2})
    if isnothing(layer.input_size) || isnothing(layer.max_indices)
        error("No input size or max indices stored. Forward pass must be called before backward pass.")
    end
    
    input_length, num_channels = layer.input_size
    grad_input = zeros(Float32, input_length, num_channels)
    
    # Szybkie rozpropagowanie gradientów
    @inbounds for c in 1:size(grad_output, 2)
        @inbounds for pos in 1:size(grad_output, 1)
            max_pos = layer.max_indices[pos, c]
            grad_input[max_pos, c] += grad_output[pos, c]
        end
    end
    
    return grad_input
end

# Obsługa wektora gradientu
function backward_pass(layer::MaxPoolLayer, grad_output::Vector{Float32})
    if ndims(grad_output) == 1
        reshaped_grad = reshape(grad_output, :, 1)
        return backward_pass(layer, reshaped_grad)
    else
        return backward_pass(layer, grad_output)
    end
end

end