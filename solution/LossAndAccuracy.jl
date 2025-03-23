module LossAndAccuracy
using Statistics: mean

export loss_and_accuracy, binary_cross_entropy_loss

function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=1))
    return exp_x ./ sum(exp_x, dims=1)
end

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

function binary_cross_entropy_loss_with_gradient(predictions, targets)
    # Apply sigmoid to get probabilities
    probabilities = sigmoid(predictions)
    
    # Calculate binary cross entropy loss
    epsilon = 1e-15  # Small value to avoid log(0)
    probabilities = clamp.(probabilities, epsilon, 1-epsilon)
    loss = -mean(targets .* log.(probabilities) .+ (1 .- targets) .* log.(1 .- probabilities))
    
    # Gradient of the loss with respect to predictions
    gradient = (probabilities .- targets) ./ size(targets, 2)
    
    return loss, Float32.(gradient)
end

function cross_entropy_loss_with_gradient(predictions, targets)
    probabilities = softmax(predictions)
    loss = -mean(sum(targets .* log.(probabilities), dims=1))
    gradient = probabilities - targets
    return loss, Float32.(gradient)
end

function one_hot_to_label(encoded)
    return [argmax(vec) for vec in eachcol(encoded)]
end

function loss_and_accuracy(ŷ, y)
    # For IMDb sentiment analysis (binary classification)
    if size(y, 1) == 2  # One-hot encoded binary classification
        loss, grad = binary_cross_entropy_loss_with_gradient(ŷ, y)
        pred_classes = [vec[1] < vec[2] ? 1 : 0 for vec in eachcol(ŷ)]
        true_classes = [vec[1] < vec[2] ? 1 : 0 for vec in eachcol(y)]
    else  # Multi-class classification (keeping for compatibility)
        loss, grad = cross_entropy_loss_with_gradient(ŷ, y)
        pred_classes = one_hot_to_label(ŷ)
        true_classes = one_hot_to_label(y)
    end
    
    acc = round(100 * mean(pred_classes .== true_classes), digits=2)
    return loss, acc, grad
end

end