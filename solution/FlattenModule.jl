module FlattenModule

export FlattenLayer, backward_pass

mutable struct FlattenLayer
    input_shape::Union{Nothing,Tuple}
    FlattenLayer() = new(nothing)
end

function (layer::FlattenLayer)(input)
    # Store the input shape for the backward pass
    layer.input_shape = size(input)
    
    # Return a column vector
    return reshape(input, :, 1)
end

function backward_pass(layer::FlattenLayer, grad_output::Array{Float32,2})
    if isnothing(layer.input_shape)
        error("Input shape must be set during the forward pass before calling backward_pass.")
    end
    
    # Reshape back to the original input shape
    return reshape(grad_output, layer.input_shape)
end

end