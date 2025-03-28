module ConvModule
using Random, LinearAlgebra

export ConvLayer, init_conv_layer, backward_pass, init_conv1d_layer

# Adnotacje dla super-szybkiej kompilacji
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

mutable struct ConvLayer
    weights::Array{Float32,3}
    biases::Array{Float32,1}
    grad_weights::Array{Float32,3}
    grad_biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Union{Nothing,Array{Float32,2}}
    # Bufory dla wydajności
    padded_input_buffer::Union{Nothing,Array{Float32,2}}
    output_buffer::Union{Nothing,Array{Float32,2}}
    pre_activation::Union{Nothing,Array{Float32,2}}
end

# Helper function dla tworzenia pre-alokowanego paddingu
@inline function create_padded_input!(padded, input, padding)
    if padding == 0
        return input
    end
    
    sequence_length, channels = size(input)
    padded_length = sequence_length + 2*padding
    
    # Wyzeruj padded - szybsza metoda
    fill!(padded, 0.0f0)
    
    # Skopiuj dane
    @inbounds for c in 1:channels
        @simd for i in 1:sequence_length
            padded[padding+i, c] = input[i, c]
        end
    end
    
    return padded
end

# Super-zoptymalizowana konwolucja 1D
@inline function fast_conv1d!(output, input, kernels, stride, padding, padded_buffer, biases)
    # Wymiary
    sequence_length, channels = size(input)
    kernel_length, input_channels, num_kernels = size(kernels)
    
    # Stwórz padded input
    if padding > 0
        padded_length = sequence_length + 2*padding
        if isnothing(padded_buffer) || size(padded_buffer, 1) != padded_length || size(padded_buffer, 2) != channels
            padded_buffer = zeros(Float32, padded_length, channels)
        end
        padded = create_padded_input!(padded_buffer, input, padding)
    else
        padded = input
    end
    
    padded_length = size(padded, 1)
    
    # Oblicz wymiary wyjścia
    out_length = div(padded_length - kernel_length, stride) + 1
    
    # Wyzeruj wyjście
    fill!(output, 0.0f0)
    
    # Konwolucja (wektoryzowana i zoptymalizowana)
    @inbounds for k in 1:num_kernels
        @inbounds for i in 1:stride:padded_length-kernel_length+1
            out_idx = div(i - 1, stride) + 1
            
            # Używamy funkcji wewnętrznej dla lepszej optymalizacji
            @inbounds for c in 1:channels
                # Ręczne rozwinięcie wewnętrznej pętli dla szybszego dostępu do pamięci
                acc = 0.0f0
                @inbounds for j in 1:kernel_length
                    acc += padded[i+j-1, c] * kernels[j, c, k]
                end
                output[out_idx, k] += acc
            end
        end
        
        # Dodaj bias (wektoryzowane)
        @inbounds for i in 1:out_length
            output[i, k] += biases[k]
        end
    end
    
    return output
end

# Szybszy operator wywołania
function (cl::ConvLayer)(input::Array{Float32,2})
    sequence_length, channels = size(input)
    cl.last_input = input
    
    # Oblicz wymiary wyjścia
    kernel_length, _, _ = size(cl.weights)  # Pobierz kernel_length z wag
    padded_length = sequence_length + 2*cl.padding
    out_length = div(padded_length - kernel_length, cl.stride) + 1
    
    # Utwórz lub użyj pre-alokowane bufory
    if isnothing(cl.output_buffer) || size(cl.output_buffer, 1) != out_length || size(cl.output_buffer, 2) != size(cl.weights, 3)
        cl.output_buffer = zeros(Float32, out_length, size(cl.weights, 3))
    end
    
    if isnothing(cl.pre_activation) || size(cl.pre_activation) != size(cl.output_buffer)
        cl.pre_activation = similar(cl.output_buffer)
    end
    
    # Wykonaj konwolucję
    fast_conv1d!(cl.output_buffer, input, cl.weights, cl.stride, cl.padding, cl.padded_input_buffer, cl.biases)
    
    # Zachowaj pre-aktywacje
    copyto!(cl.pre_activation, cl.output_buffer)
    
    # Zastosuj ReLU bezpośrednio na buforze (in-place)
    @inbounds for i in eachindex(cl.output_buffer)
        cl.output_buffer[i] = max(0.0f0, cl.output_buffer[i])
    end
    
    return cl.output_buffer
end

# Obsługa wektorów wejściowych
function (cl::ConvLayer)(input::Vector{Float32})
    reshaped_input = reshape(input, :, 1)
    return cl(reshaped_input)
end

# Inicjalizacja warstwy konwolucyjnej
function init_conv1d_layer(kernel_length::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int, seedy::Int)
    Random.seed!(seedy)
    
    # Xavier inicjalizacja
    scale = sqrt(2.0f0 / (input_channels * kernel_length + output_channels))
    weights = scale * randn(Float32, kernel_length, input_channels, output_channels)
    
    biases = zeros(Float32, output_channels)
    grad_weights = zeros(Float32, kernel_length, input_channels, output_channels)
    grad_biases = zeros(Float32, output_channels)
    
    # Pre-alokuj bufory
    padded_buffer = padding > 0 ? zeros(Float32, kernel_length + 2*padding, input_channels) : nothing
    output_buffer = nothing
    pre_activation = nothing
    
    return ConvLayer(weights, biases, grad_weights, grad_biases, stride, padding, 
                    nothing, padded_buffer, output_buffer, pre_activation)
end

# Alias dla kompatybilności
const init_conv_layer = init_conv1d_layer

# Zoptymalizowany backward pass
function backward_pass(cl::ConvLayer, grad_output::Array{Float32,2})
    if isnothing(cl.last_input) || isnothing(cl.pre_activation)
        error("No input stored. Forward pass must be called before backward pass.")
    end
    
    # Zastosuj gradient ReLU
    relu_grad = similar(grad_output)
    @inbounds for i in eachindex(grad_output)
        relu_grad[i] = cl.pre_activation[i] > 0 ? grad_output[i] : 0.0f0
    end
    
    # Parametry
    input = cl.last_input
    sequence_length, channels = size(input)
    kernel_length, input_channels, num_kernels = size(cl.weights)
    stride, padding = cl.stride, cl.padding
    
    # Inicjalizuj gradient dla wejścia
    grad_input = zeros(Float32, sequence_length, channels)
    
    # Przygotuj padded input
    if padding > 0
        padded_length = sequence_length + 2*padding
        if isnothing(cl.padded_input_buffer) || size(cl.padded_input_buffer, 1) != padded_length || size(cl.padded_input_buffer, 2) != channels
            cl.padded_input_buffer = zeros(Float32, padded_length, channels)
        end
        padded = create_padded_input!(cl.padded_input_buffer, input, padding)
    else
        padded = input
    end
    padded_length = size(padded, 1)
    
    # Oblicz gradienty bias (szybka suma)
    @inbounds for k in 1:num_kernels
        @inbounds for i in 1:size(relu_grad, 1)
            cl.grad_biases[k] += relu_grad[i, k]
        end
    end
    
    # Oblicz gradienty wag (zoptymalizowane)
    @inbounds for k in 1:num_kernels
        @inbounds for i in 1:size(relu_grad, 1)
            output_grad = relu_grad[i, k]
            in_start = (i-1) * stride + 1
            
            @inbounds for c in 1:channels
                @simd for j in 1:kernel_length
                    if in_start+j-1 <= padded_length
                        cl.grad_weights[j, c, k] += padded[in_start+j-1, c] * output_grad
                    end
                end
            end
        end
    end
    
    # Oblicz gradienty wejścia (jeśli potrzebne)
    padded_grad_input = zeros(Float32, padded_length, channels)
    
    @inbounds for k in 1:num_kernels
        @inbounds for i in 1:size(relu_grad, 1)
            output_grad = relu_grad[i, k]
            in_start = (i-1) * stride + 1
            
            @inbounds for c in 1:channels
                @simd for j in 1:kernel_length
                    pos = in_start+j-1
                    if pos <= padded_length
                        padded_grad_input[pos, c] += cl.weights[j, c, k] * output_grad
                    end
                end
            end
        end
    end
    
    # Wyodrębnij część odpowiadającą oryginalnemu wejściu
    if padding > 0
        @inbounds for c in 1:channels
            @simd for i in 1:sequence_length
                grad_input[i, c] = padded_grad_input[padding+i, c]
            end
        end
    else
        grad_input = padded_grad_input
    end
    
    return grad_input
end

# Szybkie obsługiwanie wektorów gradientu
function backward_pass(cl::ConvLayer, grad_output::Vector{Float32})
    reshaped_grad = reshape(grad_output, :, 1)
    return backward_pass(cl, reshaped_grad)
end

end