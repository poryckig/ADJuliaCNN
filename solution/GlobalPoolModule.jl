module GlobalPoolModule
using Statistics

export GlobalAveragePoolLayer, backward_pass

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

mutable struct GlobalAveragePoolLayer
    input_shape::Union{Nothing, Tuple{Int, Int}}
    output_buffer::Union{Nothing, Array{Float32, 1}}  # Bufor wyjściowy
    
    GlobalAveragePoolLayer() = new(nothing, nothing)
end

# Zoptymalizowany pooling średniej globalnej
@inline function fast_global_avg_pool!(output, input)
    seq_length, channels = size(input)
    
    # Zoptymalizowana redukcja wzdłuż wymiaru sekwencji
    @inbounds for c in 1:channels
        sum_val = 0.0f0
        @inbounds for i in 1:seq_length
            sum_val += input[i, c]
        end
        output[c] = sum_val / seq_length
    end
    
    return output
end

# Efektywny operator wywołania
function (layer::GlobalAveragePoolLayer)(input::Array{Float32,2})
    # Przechowaj kształt wejścia
    layer.input_shape = size(input)
    seq_length, channels = layer.input_shape
    
    # Przygotuj bufor wyjściowy
    if isnothing(layer.output_buffer) || length(layer.output_buffer) != channels
        layer.output_buffer = zeros(Float32, channels)
    end
    
    # Wykonaj pooling
    fast_global_avg_pool!(layer.output_buffer, input)
    
    return layer.output_buffer
end

# Szybki backward pass
function backward_pass(layer::GlobalAveragePoolLayer, grad_output)
    if isnothing(layer.input_shape)
        error("No input shape stored. Forward pass must be called before backward pass.")
    end
    
    seq_length, num_channels = layer.input_shape
    
    # Konwersja do Float32 dla spójności
    grad_output_float = Float32.(grad_output)
    
    # Przygotuj gradient jako kolumnowy wektor
    if ndims(grad_output_float) == 1
        grad_vec = grad_output_float
    else
        grad_vec = vec(grad_output_float)
    end
    
    # Przygotuj wyjściowy gradient
    grad_input = zeros(Float32, seq_length, num_channels)
    
    # Oblicz skalę
    scale = 1.0f0 / seq_length
    
    # Efektywnie rozpropaguj gradient
    @inbounds for c in 1:num_channels
        grad_val = grad_vec[c] * scale
        @inbounds for i in 1:seq_length
            grad_input[i, c] = grad_val
        end
    end
    
    return grad_input
end

end