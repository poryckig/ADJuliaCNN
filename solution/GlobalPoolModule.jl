module GlobalPoolModule
using Statistics  # Add this import to use the mean function

export GlobalAveragePoolLayer, backward_pass

mutable struct GlobalAveragePoolLayer
    input_shape::Union{Nothing, Tuple{Int, Int}}
    
    GlobalAveragePoolLayer() = new(nothing)
end

function (layer::GlobalAveragePoolLayer)(input::Array{Float32,2})
    # Store input shape for backward pass
    layer.input_shape = size(input)
    
    # Average across the sequence dimension (dim 1)
    output = mean(input, dims=1)
    
    # Make it a column vector
    return reshape(output', :, 1)
end

function backward_pass(layer::GlobalAveragePoolLayer, grad_output::Array{Float32,2})
    if isnothing(layer.input_shape)
        error("No input shape stored. Forward pass must be called before backward pass.")
    end
    
    seq_length, num_channels = layer.input_shape
    
    # Reshape grad_output to (1, channels)
    grad_reshaped = reshape(grad_output, 1, :)
    
    # Repeat gradient across sequence dimension and normalize
    return repeat(grad_reshaped, outer=[seq_length, 1]) ./ seq_length
end

end