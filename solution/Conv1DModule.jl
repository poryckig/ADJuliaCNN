module Conv1DModule
using Random

mutable struct Conv1DLayer
    weights::Array{Float32,3}     # Changed from 4D to 3D (filter_length, input_channels, output_channels)
    biases::Array{Float32,1}
    grad_weights::Array{Float32,3}
    grad_biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Union{Nothing,Array{Float32,2}}  # Changed from 3D to 2D (sequence_length, channels)
end

function forward(input::Array{Float32,2}, kernels::Array{Float32,3}, stride::Int, padding::Int)
    # Input dimensions
    (sequence_length, channels) = size(input)

    # Kernel dimensions
    (kernel_length, _, num_kernels) = size(kernels)

    # Output dimensions
    out_length = div(sequence_length - kernel_length + 2 * padding, stride) + 1

    # Apply padding
    padded_input = zeros(Float32, sequence_length + 2 * padding, channels)
    padded_input[padding+1:padding+sequence_length, :] = input

    # Output initialization
    output = zeros(Float32, out_length, num_kernels)

    # Perform 1D convolution for each filter
    for k in 1:num_kernels
        kernel = kernels[:, :, k]  # kernel is now (kernel_length, channels)
        for pos in 1:stride:sequence_length-kernel_length+1+2*padding
            patch = padded_input[pos:pos+kernel_length-1, :]
            output[div(pos - 1, stride)+1, k] += sum(patch .* kernel)
        end
    end

    return output
end

function (cl::Conv1DLayer)(input::Array{Float32,2})
    cl.last_input = copy(input)  # Store the original input for backward pass

    # Perform 1D convolution
    conv_output = forward(input, cl.weights, cl.stride, cl.padding)

    # Add bias (broadcasting addition across filters)
    for c in axes(conv_output, 2)
        conv_output[:, c] .+= cl.biases[c]
    end

    # Apply ReLU activation function
    return relu(conv_output)
end

# Implementing ReLU activation function
function relu(x)
    return max.(0, x)
end

function init_conv1d_layer(kernel_length::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int, seedy::Int)
    Random.seed!(seedy)

    weights = 0.178 * randn(Float32, kernel_length, input_channels, output_channels)
    biases = zeros(Float32, output_channels)
    grad_weights = zeros(Float32, kernel_length, input_channels, output_channels)
    grad_biases = zeros(Float32, output_channels)
    return Conv1DLayer(weights, biases, grad_weights, grad_biases, stride, padding, nothing)
end

function backward_pass(cl::Conv1DLayer, grad_output::Array{Float32,2})
    input = cl.last_input
    (sequence_length, channels) = size(input)
    (kernel_length, _, num_kernels) = size(cl.weights)
    stride, padding = cl.stride, cl.padding

    grad_input = zeros(Float32, size(input))
    
    # Prepare padded input for computations
    padded_input = zeros(Float32, sequence_length + 2 * padding, channels)
    padded_input[padding+1:end-padding, :] = input

    for k in 1:num_kernels
        for pos in 1:stride:sequence_length-kernel_length+1+2*padding
            pos_out = div(pos - 1, stride) + 1
            if pos_out <= size(grad_output, 1)
                patch = padded_input[pos:pos+kernel_length-1, :]
                grad_bias = grad_output[pos_out, k]
                cl.grad_biases[k] += grad_bias
                cl.grad_weights[:, :, k] += patch * grad_bias
            end
        end
    end

    return Float32.(grad_input)
end

end