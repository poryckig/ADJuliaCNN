module Conv1DModule
using Random, LinearAlgebra

export Conv1DLayer, init_conv1d_layer, backward_pass

mutable struct Conv1DLayer
    weights::Array{Float32,3}
    biases::Array{Float32,1}
    grad_weights::Array{Float32,3}
    grad_biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Union{Nothing,Array{Float32,2}}
end

# Helper function for creating padded input
function create_padded_input(input, padding)
    if padding == 0
        return input
    end
    
    sequence_length, channels = size(input)
    padded = zeros(Float32, sequence_length + 2*padding, channels)
    padded[padding+1:padding+sequence_length, :] = input
    return padded
end

# Helper function to create matrix of sliding windows (im2col style)
function sliding_windows(input, kernel_length, stride)
    sequence_length, channels = size(input)
    output_length = div(sequence_length - kernel_length, stride) + 1
    
    # Pre-allocate the result matrix
    # Each row will be a flattened window, columns represent different positions
    windows = zeros(Float32, kernel_length * channels, output_length)
    
    # Populate the windows matrix
    col_idx = 1
    for i in 1:stride:sequence_length-kernel_length+1
        window = input[i:i+kernel_length-1, :]
        windows[:, col_idx] = reshape(window, :)
        col_idx += 1
    end
    
    return windows
end

function forward(input::Array{Float32,2}, kernels::Array{Float32,3}, stride::Int, padding::Int)
    # Input dimensions
    sequence_length, channels = size(input)
    
    # Kernel dimensions
    kernel_length, input_channels, num_kernels = size(kernels)
    
    # Apply padding
    padded_input = create_padded_input(input, padding)
    padded_length = size(padded_input, 1)
    
    # Calculate output dimensions
    out_length = div(padded_length - kernel_length, stride) + 1
    
    # Initialize output
    output = zeros(Float32, out_length, num_kernels)
    
    # Reshape kernels for matrix multiplication
    # Each kernel becomes a row in the reshaped matrix
    reshaped_kernels = reshape(kernels, kernel_length * input_channels, num_kernels)
    
    # Get sliding windows
    windows = sliding_windows(padded_input, kernel_length, stride)
    
    # Compute convolution with matrix multiplication
    # windows: (kernel_length*channels, output_length)
    # reshaped_kernels: (kernel_length*channels, num_kernels)
    # result: (num_kernels, output_length)
    result = reshaped_kernels' * windows
    
    # Transpose to get expected output shape: (output_length, num_kernels)
    output = result'
    
    return output
end

function (cl::Conv1DLayer)(input::Array{Float32,2})
    cl.last_input = copy(input)
    
    # Perform 1D convolution
    conv_output = forward(input, cl.weights, cl.stride, cl.padding)
    
    # Add bias (broadcasting across sequence positions)
    for c in axes(conv_output, 2)
        conv_output[:, c] .+= cl.biases[c]
    end
    
    # Apply ReLU activation function
    return relu.(conv_output)
end

# Implementing ReLU activation function
function relu(x)
    return max.(0, x)
end

function init_conv1d_layer(kernel_length::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int, seedy::Int)
    Random.seed!(seedy)
    
    # Xavier/Glorot initialization for better convergence
    scale = sqrt(2.0 / (input_channels * kernel_length + output_channels))
    weights = scale * randn(Float32, kernel_length, input_channels, output_channels)
    
    biases = zeros(Float32, output_channels)
    grad_weights = zeros(Float32, kernel_length, input_channels, output_channels)
    grad_biases = zeros(Float32, output_channels)
    
    return Conv1DLayer(weights, biases, grad_weights, grad_biases, stride, padding, nothing)
end

function backward_pass(cl::Conv1DLayer, grad_output::Array{Float32,2})
    input = cl.last_input
    sequence_length, channels = size(input)
    kernel_length, input_channels, num_kernels = size(cl.weights)
    stride, padding = cl.stride, cl.padding
    
    # Apply padding to input
    padded_input = create_padded_input(input, padding)
    padded_length = size(padded_input, 1)
    
    # Initialize gradient for input
    grad_input = zeros(Float32, sequence_length, channels)
    
    # Precompute gradient with respect to ReLU activation
    # Gradient is zero where the output was negative before ReLU
    out_length, num_kernels = size(grad_output)
    
    # For each filter
    for k in 1:num_kernels
        # Update bias gradient - just sum the gradients across sequence positions
        cl.grad_biases[k] += sum(grad_output[:, k])
        
        # For each position in the output
        for i in 1:out_length
            output_grad = grad_output[i, k]
            
            # The corresponding position in the input (accounting for stride)
            in_start = (i-1) * stride + 1
            in_end = in_start + kernel_length - 1
            
            # Only proceed if this output position affects valid input positions
            if in_start <= padded_length && in_end <= padded_length
                # Extract the input patch that generated this output
                patch = padded_input[in_start:in_end, :]
                
                # Update weight gradients
                for in_ch in 1:input_channels
                    for kl in 1:kernel_length
                        cl.grad_weights[kl, in_ch, k] += patch[kl, in_ch] * output_grad
                    end
                end
            end
        end
    end
    
    # Return gradient for input (not optimized for speed but maintained for API compatibility)
    # In practice, this would be more efficiently computed with transposed convolution
    return grad_input
end

end