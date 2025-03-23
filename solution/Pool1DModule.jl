module Pool1DModule

export MaxPool1DLayer, init_pool1d_layer, apply_pooling, maxpool_backward

mutable struct MaxPool1DLayer
    pool_length::Int    # Changed from 2D (height/width) to 1D (length)
    stride::Int
    max_indices::Union{Nothing,Array{Int,2}}  # Changed from 3D to 2D (position, channel)
end

function init_pool1d_layer(pool_length::Int, stride::Int)
    MaxPool1DLayer(pool_length, stride, nothing)
end

function apply_pooling(layer::MaxPool1DLayer, input::Array{Float32,2})
    (input_length, num_channels) = size(input)
    output_length = div(input_length - layer.pool_length, layer.stride) + 1

    output = zeros(Float32, output_length, num_channels)
    layer.max_indices = Array{Int,2}(undef, output_length, num_channels)

    for c in 1:num_channels
        for pos in 1:layer.stride:input_length-layer.pool_length+1
            window = input[pos:pos+layer.pool_length-1, c]
            max_value = maximum(window)
            output_idx = div(pos - 1, layer.stride) + 1
            output[output_idx, c] = max_value
            idx = findfirst(isequal(max_value), window)
            layer.max_indices[output_idx, c] = pos + idx - 1
        end
    end

    return output, layer.max_indices
end

function (layer::MaxPool1DLayer)(input::Array{Float32,2})
    output, _ = apply_pooling(layer, input)
    return output
end

function backward_pass(layer::MaxPool1DLayer, grad_output::Array{Float32,2})
    grad_input = zeros(Float32, calculate_input_dimensions(layer, size(grad_output)...))
    for c in 1:size(grad_output, 2)
        for pos in 1:size(grad_output, 1)
            max_pos = layer.max_indices[pos, c]
            grad_input[max_pos, c] += grad_output[pos, c]
        end
    end

    return grad_input
end

function calculate_input_dimensions(layer::MaxPool1DLayer, out_length::Int, num_channels::Int)
    input_length = out_length * layer.stride + layer.pool_length - 1
    return (input_length, num_channels)
end

end