module IMDbDataLoader

using Random, HTTP, JSON, LinearAlgebra
export load_data, preprocess_data, one_hot_encode, batch_data

# Constants
const IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
const VOCAB_SIZE = 10000
const MAX_SEQUENCE_LENGTH = 100  # Reduced for performance

function download_imdb_dataset()
    data_dir = joinpath(@__DIR__, "data", "imdb")
    
    if !isdir(data_dir)
        println("IMDb dataset not found. Downloading...")
        mkpath(data_dir)
        
        # Download the dataset
        tarball = HTTP.get(IMDB_URL).body
        
        # Save the tarball
        tarball_path = joinpath(data_dir, "imdb.tar.gz")
        open(tarball_path, "w") do f
            write(f, tarball)
        end
        
        # Extract tarball
        run(`tar -xzf $tarball_path -C $data_dir`)
        
        println("Dataset downloaded and extracted.")
    end
    
    return data_dir
end

function build_vocabulary(data_dir, max_words=VOCAB_SIZE)
    println("Building vocabulary...")
    # Collect all text files
    all_texts = String[]
    
    for split in ["train", "test"]
        for sentiment in ["pos", "neg"]
            dir_path = joinpath(data_dir, "aclImdb", split, sentiment)
            if isdir(dir_path)
                for file in readdir(dir_path)
                    if endswith(file, ".txt")
                        push!(all_texts, read(joinpath(dir_path, file), String))
                    end
                end
            end
        end
    end
    
    # Tokenize and count word frequencies
    word_counts = Dict{String, Int}()
    for text in all_texts
        words = split(lowercase(text))
        for word in words
            word_counts[word] = get(word_counts, word, 0) + 1
        end
    end
    
    # Sort words by frequency and take top max_words
    sorted_words = sort(collect(word_counts), by=x->x[2], rev=true)
    vocabulary = Dict{String, Int}()
    
    # Add special tokens
    vocabulary["<PAD>"] = 1
    vocabulary["<UNK>"] = 2
    
    # Add top words
    for (i, (word, _)) in enumerate(sorted_words[1:min(max_words-2, length(sorted_words))])
        vocabulary[word] = i + 2
    end
    
    return vocabulary
end

function load_data(split::Symbol)
    println("Processing $(split) data...")
    # Load or download dataset
    data_dir = download_imdb_dataset()
    
    # Build or load vocabulary
    vocab_file = joinpath(data_dir, "vocabulary.json")
    if isfile(vocab_file)
        vocabulary = JSON.parse(read(vocab_file, String))
    else
        vocabulary = build_vocabulary(data_dir)
        open(vocab_file, "w") do f
            write(f, JSON.json(vocabulary))
        end
    end
    
    # Determine paths
    if split == :train
        pos_dir = joinpath(data_dir, "aclImdb", "train", "pos")
        neg_dir = joinpath(data_dir, "aclImdb", "train", "neg")
    elseif split == :test
        pos_dir = joinpath(data_dir, "aclImdb", "test", "pos")
        neg_dir = joinpath(data_dir, "aclImdb", "test", "neg")
    else
        error("Invalid split: $split. Use :train or :test")
    end
    
    # Load positive and negative reviews
    features = []
    targets = []
    
    # Process positive reviews
    for file in readdir(pos_dir)
        if endswith(file, ".txt")
            text = read(joinpath(pos_dir, file), String)
            tokens = tokenize_and_pad(text, vocabulary)
            push!(features, tokens)
            push!(targets, 1)  # Positive sentiment
        end
    end
    
    # Process negative reviews
    for file in readdir(neg_dir)
        if endswith(file, ".txt")
            text = read(joinpath(neg_dir, file), String)
            tokens = tokenize_and_pad(text, vocabulary)
            push!(features, tokens)
            push!(targets, 0)  # Negative sentiment
        end
    end
    
    return features, targets
end

function tokenize_and_pad(text, vocabulary, max_length=MAX_SEQUENCE_LENGTH)
    words = split(lowercase(text))
    tokens = zeros(Int, max_length)
    
    for (i, word) in enumerate(words[1:min(length(words), max_length)])
        if haskey(vocabulary, word)
            tokens[i] = vocabulary[word]
        else
            tokens[i] = vocabulary["<UNK>"]
        end
    end
    
    return tokens
end

function preprocess_data(features, targets; one_hot::Bool=true)
    # Convert features to Float32 arrays
    x = [Float32.(feature) for feature in features]
    
    # One-hot encode targets if requested
    y = one_hot ? one_hot_encode(targets, 0:1) : targets
    
    return x, y
end

function one_hot_encode(targets, classes)
    one_hot = zeros(Float32, length(classes), length(targets))
    for (i, class) in enumerate(classes)
        filter_indices = findall(x -> x == class, targets)
        one_hot[i, filter_indices] .= 1
    end
    return one_hot
end

# Funkcja do rozszerzania danych
function augment_data(tokens, vocabulary, max_length=MAX_SEQUENCE_LENGTH, dropout_prob=0.1)
    # Kopia tokenu
    augmented = copy(tokens)
    
    # Losowe usuwanie tokenów (dropout)
    for i in 1:length(augmented)
        if augmented[i] > 0 && rand() < dropout_prob
            augmented[i] = vocabulary["<UNK>"]
        end
    end
    
    return augmented
end

# Zaktualizuj funkcję batch_data, aby obsługiwała augmentację
function batch_data(data, batch_size::Int; shuffle::Bool=true, augment::Bool=true)
    x, y = data
    indices = 1:length(x)
    if shuffle
        indices = Random.shuffle(indices)
    end
    
    # Przygotuj słownik dla augmentacji (jeśli potrzebne)
    vocab = Dict("<UNK>" => 2)
    
    # Utwórz partie sekwencji i etykiet
    batches = []
    for idx_batch in Iterators.partition(indices, batch_size)
        # Przygotuj oryginalny batch
        x_batch = [x[i] for i in idx_batch]
        y_batch = y[:, idx_batch]
        
        # Jeśli augment jest włączony i jest to trening
        if augment
            # Rozszerz dane
            x_augmented = [augment_data(xi, vocab) for xi in x_batch]
            push!(batches, (x_augmented, y_batch))
        else
            push!(batches, (x_batch, y_batch))
        end
    end
    
    return batches
end

end