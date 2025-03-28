module LossAndAccuracy
using Statistics: mean

export loss_and_accuracy, binary_cross_entropy_loss

# Super-szybka kompilacja
Base.Experimental.@optlevel 3
Base.@propagate_inbounds true

# Ultraszybki softmax
@inline function fast_softmax!(output, input)
    # Stabilizacja numeryczna
    max_val = maximum(input)
    sum_exp = 0.0f0
    
    # Oblicz eksponenty i sumę
    @inbounds for i in eachindex(input)
        output[i] = exp(input[i] - max_val)
        sum_exp += output[i]
    end
    
    # Normalizuj
    inv_sum = 1.0f0 / sum_exp
    @inbounds for i in eachindex(output)
        output[i] *= inv_sum
    end
    
    return output
end

# Szybki i numerycznie stabilny sigmoid
@inline function fast_sigmoid(x::Float32)
    if x >= 0
        t = exp(-x)
        return 1.0f0 / (1.0f0 + t)
    else
        t = exp(x)
        return t / (1.0f0 + t)
    end
end

@inline function fast_sigmoid!(output, input)
    @inbounds for i in eachindex(input)
        output[i] = fast_sigmoid(input[i])
    end
    return output
end

# Zoptymalizowany cross entropy loss z gradientem
@inline function binary_cross_entropy_loss_with_gradient!(loss, gradient, predictions, targets, lambda=0.0001f0, network=nothing)
    epsilon = Float32(1e-7)  # Mała wartość dla stabilności
    batch_size = size(targets, 2)
    
    # Tymczasowe bufory dla prawdopodobieństw
    probs = similar(predictions)
    fast_sigmoid!(probs, predictions)
    
    # Oblicz straty i gradienty
    total_loss = 0.0f0
    
    @inbounds for j in 1:size(targets, 2)
        @inbounds for i in 1:size(targets, 1)
            # Przycinanie dla stabilności numerycznej
            p = max(min(probs[i, j], 1.0f0 - epsilon), epsilon)
            t = targets[i, j]
            
            # Strata
            total_loss -= t * log(p) + (1.0f0 - t) * log(1.0f0 - p)
            
            # Gradient
            gradient[i, j] = p - t
        end
    end
    
    # Dodaj regularyzację L2 jeśli sieć jest dostępna
    if network !== nothing && lambda > 0.0f0
        reg_loss = 0.0f0
        for layer in network
            if hasproperty(layer, :weights)
                reg_loss += 0.5f0 * lambda * sum(layer.weights.^2)
            end
        end
        total_loss += reg_loss
    end
    
    # Średnia strata
    loss[1] = total_loss / batch_size
    
    return loss[1], gradient
end

# Uniwersalna funkcja do obliczania straty i dokładności
function loss_and_accuracy(ŷ, y, network=nothing)
    batch_size = size(y, 2)
    
    # Przygotuj bufory
    loss_buffer = [0.0f0]
    grad_buffer = similar(ŷ)
    
    # Dla klasyfikacji binarnej (IMDb sentiment)
    if size(y, 1) == 2
        # Oblicz stratę i gradienty z regularyzacją
        loss, grad = binary_cross_entropy_loss_with_gradient!(loss_buffer, grad_buffer, ŷ, y, 0.0001f0, network)
        
        # Przewidywana klasa
        pred_probs = similar(ŷ)
        fast_sigmoid!(pred_probs, ŷ)
        
        # Oblicz dokładność
        correct = 0
        @inbounds for j in 1:size(y, 2)
            pred_class = pred_probs[2, j] > 0.5f0 ? 1 : 0
            true_class = y[2, j] > y[1, j] ? 1 : 0
            correct += pred_class == true_class ? 1 : 0
        end
        
        accuracy = correct / batch_size
    else
        # Dla innych przypadków (kompatybilność)
        error("Only binary classification supported in this optimized version")
    end
    
    return loss, accuracy, grad_buffer
end

end