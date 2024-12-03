using CSV, DataFrames, Statistics, Dates, Gadfly, Combinatorics, Plots, StatsBase, StatsPlots, Random, StatsModels, GLM, LinearAlgebra, MultivariateStats, Distributions

full_train = CSV.read("./data/raw/train.csv", DataFrame; delim=";")
test =  CSV.read("./data/raw/test.csv", DataFrame; delim=";") #ne contient pas la varialbe consommation

Random.seed!(1234) #pour la reproductibilit

ntrain = round(Int, .8*nrow(full_train)) #80% des données pour l'entrainement: 80% * nb de lignes

train_id = sample(1:nrow(full_train), ntrain, replace=false, ordered=true) #échantillonnage aléatoire pour l'entrainement
valid_id = setdiff(1:nrow(full_train), train_id) #échantillon de validation. prend celles qui ne sont pas dans l'échantillon d'entrainement

train = full_train[train_id, :]  
valid = full_train[valid_id, :]

function safe_parse_float(x)
    try
        return parse(Float64, x)
    catch
        return missing
    end
end

function one_hot_encode(df, cols, levels_dict)
    for col in cols
        levels_col = levels_dict[col]
        for level in levels_col
            new_col = Symbol(string(col) * "_" * string(level))
            df[!, new_col] = ifelse.(df[!, col] .== level, 1.0, 0.0)
        end
        select!(df, Not(col))
    end
    return df
end

## convert annee column into age
train.age = 2024 .- train.annee
valid.age = 2024 .- valid.annee
test.age = 2024 .- test.annee

train = select!(train, Not(:annee))
valid = select!(valid, Not(:annee))
test = select!(test, Not(:annee))

## drop missing values
train = dropmissing(train)
valid = dropmissing(valid)
test = dropmissing(test)

# Datasets that contain 'consommation'
datasets_with_consommation = [train, valid]

# Datasets without 'consommation'
datasets_without_consommation = [test]

# Apply replacements to 'cylindree' in all datasets
for df in [train, valid, test]
    df.cylindree = replace.(df.cylindree, "," => ".")
end

# Apply replacements to 'consommation' only in datasets that have it
for df in datasets_with_consommation
    df.consommation = replace.(df.consommation, "," => ".")
end

# Convert 'cylindree' to float in all datasets
for df in [train, valid, test]
    df.cylindree = safe_parse_float.(df.cylindree)
end

# Convert 'consommation' to float in datasets with 'consommation'
for df in datasets_with_consommation
    df.consommation = safe_parse_float.(df.consommation)
end

# Drop missing values in all datasets
for df in [train, valid, test]
    dropmissing!(df)
end

# Define categorical columns
categorical_cols = [:type, :transmission, :boite]

# Collect unique levels from the training set
levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
end

train = one_hot_encode(train, categorical_cols, levels_dict)
valid = one_hot_encode(valid, categorical_cols, levels_dict)
test = one_hot_encode(test, categorical_cols, levels_dict)

y_train = train.consommation
X_train = select(train, Not(:consommation))
y_valid = valid.consommation
X_valid = select(valid, Not(:consommation))
X_test = deepcopy(test)


# Identify numeric feature indices
feature_names = names(train)
numeric_features = [ :cylindree, :nombre_cylindres, :age]
numeric_indices = findall(x -> x in numeric_features, feature_names)

means = mean(Matrix(X_train[:, numeric_features]), dims=1)
stds = std(Matrix(X_train[:, numeric_features]), dims=1)

function standardizer(X, means, stds)
    X = deepcopy(X)
    for j in 1:size(X, 2)
        if j in numeric_indices
            X[:, j] = (X[:, j] .- means[j]) ./ stds[j]
        end
    end
    return X
end

X_train = standardizer(Matrix(X_train), means, stds)
X_valid = standardizer(Matrix(X_valid), means, stds)
X_test = standardizer(Matrix(X_test), means, stds)

y_train = Vector(y_train)
y_valid = Vector(y_valid)

# Ridge regression with cross-validation
XtX = X_train' * X_train
Xty = X_train' * y_train
n_features = size(X_train, 2)

lambda_values = 10 .^ range(-5, stop=5, length=1000)
best_rmse = Inf
best_lambda = 0.0
best_beta = nothing

for λ in lambda_values
    beta = (XtX + λ * I) \ Xty
    y_pred_valid = X_valid * beta
    rmse = sqrt(mean((y_pred_valid - y_valid).^2))
    global best_rmse, best_lambda, best_beta
    if rmse < best_rmse
        best_rmse = rmse
        best_lambda = λ
        best_beta = beta
    end
end

println("Best Lambda: ", best_lambda)
println("Best RMSE: ", best_rmse)

# Evaluation on validation set
y_valid_pred = X_valid * best_beta
rmse_valid = sqrt(mean((y_valid_pred - y_valid).^2))
println("Validation RMSE: ", rmse_valid)

# cross validation 
# Prepare DataFrame for cross-validation
n_folds = 5
n_samples = size(X_train, 1)
fold_size = n_samples ÷ n_folds
folds = repeat(1:n_folds, inner=fold_size)
shuffle!(folds)

# Cross-validation
rmse_values = []
for λ in lambda_values
    rmse = 0.0
    for fold in 1:n_folds
        train_indices = findall(x -> x != fold, folds)
        valid_indices = findall(x -> x == fold, folds)
        X_train_fold = X_train[train_indices, :]
        y_train_fold = y_train[train_indices]
        X_valid_fold = X_train[valid_indices, :]
        y_valid_fold = y_train[valid_indices]
        beta = (X_train_fold' * X_train_fold + λ * I) \ (X_train_fold' * y_train_fold)
        y_pred_valid = X_valid_fold * beta
        rmse += sqrt(mean((y_pred_valid - y_valid_fold).^2))
    end
    push!(rmse_values, rmse / n_folds)
end

# Plot RMSE values
Gadfly.plot(x=lambda_values, y=rmse_values, Geom.line, Guide.xlabel("λ"), Guide.ylabel("RMSE"))

# Find best λ
best_lambda = lambda_values[argmin(rmse_values)]
println("Best λ: ", best_lambda)

# Train model with best λ
beta = (X_train' * X_train + best_lambda * I) \ (X_train' * y_train)

# Evaluate on validation set
y_valid_pred = X_valid * beta
rmse_valid = sqrt(mean((y_valid_pred - y_valid).^2))
println("Validation RMSE: ", rmse_valid)





# # Predictions on test set
# y_test_pred = X_test * best_beta

# # Prepare submission DataFrame
# n_test = size(y_test_pred, 1)
# id = 1:n_test
# df_pred = DataFrame(id=id, consommation=y_test_pred)

# name = "ridge" * string(rmse_valid) * ".csv"
# CSV.write("../submissions/" * name, df_pred)
# println("Predictions exported successfully to " * name * ".")