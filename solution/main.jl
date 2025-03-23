using Statistics

include("Conv1DModule.jl")
include("Pool1DModule.jl")
include("FlattenModule.jl")
include("DenseModule.jl")
include("EmbeddingModule.jl")

include("IMDbDataLoader.jl")
include("LossAndAccuracy.jl")
include("NetworkHandlers.jl")

using .Conv1DModule, .Pool1DModule, .IMDbDataLoader, .FlattenModule, .DenseModule, .EmbeddingModule

# Load and preprocess the data
train_features, train_labels = IMDbDataLoader.load_data(:train)
train_x, train_y = IMDbDataLoader.preprocess_data(train_features, train_labels; one_hot=true)

# Load and preprocess test data
test_features, test_labels = IMDbDataLoader.load_data(:test)
test_x, test_y = IMDbDataLoader.preprocess_data(test_features, test_labels; one_hot=true)

# Create batches
batch_size = 32
train_data = IMDbDataLoader.batch_data((train_x, train_y), batch_size; shuffle=true)

# Parameters
vocab_size = 10000
embedding_dim = 100
sequence_length = 500  # Maximum review length

# Initialize embedding and convolutional layers
embedding_layer = EmbeddingModule.init_embedding_layer(vocab_size, embedding_dim, 3697631579)
conv_layer1 = Conv1DModule.init_conv1d_layer(3, embedding_dim, 64, 1, 0, 3697631579)
pool_layer1 = Pool1DModule.init_pool1d_layer(2, 2)
conv_layer2 = Conv1DModule.init_conv1d_layer(3, 64, 128, 1, 0, 3731614026)
pool_layer2 = Pool1DModule.init_pool1d_layer(2, 2)
flatten_layer = FlattenModule.FlattenLayer()

# Calculate the expected output size after convolutions and pooling
# Use a sample input to determine dimensions dynamically
sample_input = train_x[1]
sample_embedded = embedding_layer(sample_input)
sample_conv1 = conv_layer1(sample_embedded)
sample_pool1 = pool_layer1(sample_conv1)
sample_conv2 = conv_layer2(sample_pool1)
sample_pool2 = pool_layer2(sample_conv2)
sample_flat = flatten_layer(sample_pool2)

# Get the actual flattened size
flattened_size = size(sample_flat, 1)
println("Flattened output size: ", flattened_size)

# Initialize dense layers with the correct input dimension
dense_layer1 = DenseModule.init_dense_layer(flattened_size, 64, DenseModule.relu, DenseModule.relu_grad, 4172219205)
dense_layer2 = DenseModule.init_dense_layer(64, 2, DenseModule.sigmoid, DenseModule.sigmoid_grad, 3762133366)

# Assemble network with dynamically calculated dimensions
network = (embedding_layer, conv_layer1, pool_layer1, conv_layer2, pool_layer2, flatten_layer, dense_layer1, dense_layer2)

# Backward pass and update functions (unchanged)
function backward_pass_master(network, grad_loss)
    for layer in reverse(network)
        if isa(layer, Conv1DModule.Conv1DLayer)
            grad_loss = Conv1DModule.backward_pass(layer, grad_loss)
        elseif isa(layer, Pool1DModule.MaxPool1DLayer)
            grad_loss = Pool1DModule.backward_pass(layer, grad_loss)
        elseif isa(layer, DenseModule.DenseLayer)
            grad_loss = DenseModule.backward_pass(layer, grad_loss)
        elseif isa(layer, FlattenModule.FlattenLayer)
            grad_loss = FlattenModule.backward_pass(layer, grad_loss)
        elseif isa(layer, EmbeddingModule.EmbeddingLayer)
            grad_loss = EmbeddingModule.backward_pass(layer, grad_loss)
        else
            println("No backward pass defined for layer type $(typeof(layer))")
        end
    end
    return grad_loss
end

function update_weights(network, learning_rate)
    for layer in reverse(network)
        if isa(layer, DenseModule.DenseLayer) || 
           isa(layer, Conv1DModule.Conv1DLayer) || 
           isa(layer, EmbeddingModule.EmbeddingLayer)
            
            layer.grad_weights ./= batch_size
            layer.grad_biases ./= batch_size
            
            layer.weights .-= learning_rate * layer.grad_weights
            layer.biases .-= learning_rate * layer.grad_biases
            
            fill!(layer.grad_weights, 0)
            fill!(layer.grad_biases, 0)
        end
    end
end

# Evaluation function
function evaluate_model(network, test_x, test_y)
    total_loss = 0.0
    total_accuracy = 0.0
    num_samples = min(length(test_x), 100)  # Evaluate on a subset to save time
    
    for i in 1:num_samples
        input = test_x[i]
        target = test_y[:, i]
        
        # Forward pass
        output = NetworkHandlers.forward_pass_master(network, input)
        
        # Calculate loss and accuracy
        loss, accuracy, _ = LossAndAccuracy.loss_and_accuracy(output, target)
        total_loss += loss
        total_accuracy += accuracy
    end
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / num_samples
    avg_accuracy = total_accuracy / num_samples
    return avg_loss, avg_accuracy
end

# Training loop
using .NetworkHandlers, .LossAndAccuracy
epochs = 2
training_step = 0.001

println("Starting training...")

for epoch in 1:epochs
    accumulated_accuracy_epoch = 0.0
    samples_processed = 0
    
    for (batch_idx, batch) in enumerate(train_data)
        batch_inputs, batch_targets = batch
        batch_loss = 0.0
        batch_accuracy = 0.0
        
        for i in 1:length(batch_inputs)
            input = batch_inputs[i]
            target = batch_targets[:, i]
            
            output = NetworkHandlers.forward_pass_master(network, input)
            
            loss, accuracy, grad_loss = LossAndAccuracy.loss_and_accuracy(output, target)
            batch_loss += loss
            batch_accuracy += accuracy
            accumulated_accuracy_epoch += accuracy
            
            backward_pass_master(network, grad_loss)
        end
        
        samples_processed += length(batch_inputs)
        
        # Update weights after each batch
        update_weights(network, training_step)
        
        if batch_idx % 5 == 0
            println("Epoch $(epoch), Batch $(batch_idx), Loss: $(batch_loss/length(batch_inputs)), Accuracy: $(batch_accuracy/length(batch_inputs))")
        end
    end
    
    # Evaluate after each epoch
    test_loss, test_accuracy = evaluate_model(network, test_x, test_y)
    
    println("Epoch $(epoch) completed:")
    println("  Training Accuracy: $(accumulated_accuracy_epoch / samples_processed)")
    println("  Test Accuracy: $(test_accuracy)")
end