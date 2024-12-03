using CSV, DataFrames, Statistics, Dates, Gadfly, Combinatorics, Plots, StatsBase, StatsPlots, Random, StatsModels, GLM, LinearAlgebra, MultivariateStats, Distributions

using MultivariateStats
using GLM

full_train = CSV.read("./data/raw/train.csv", DataFrame; delim=";")
test =  CSV.read("./data/raw/test.csv", DataFrame; delim=";") #ne contient pas la varialbe consommation
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

Random.seed!(1234) #pour la reproductibilit

ntrain = round(Int, .8*nrow(full_train)) #80% des données pour l'entrainement: 80% * nb de lignes

train_id = sample(1:nrow(full_train), ntrain, replace=false, ordered=true) #échantillonnage aléatoire pour l'entrainement
valid_id = setdiff(1:nrow(full_train), train_id) #échantillon de validation. prend celles qui ne sont pas dans l'échantillon d'entrainement

train = full_train[train_id, :]  
valid = full_train[valid_id, :]


# Datasets that contain 'consommation'
datasets_with_consommation = [train, valid]

# Datasets without 'consommation'
datasets_without_consommation = [test]

# Data Cleaning and Preparation
Random.seed!(1234) # For reproducibility

# Split the data
ntrain = round(Int, 0.8 * nrow(full_train))
train_id = sample(1:nrow(full_train), ntrain; replace=false, ordered=true)
valid_id = setdiff(1:nrow(full_train), train_id)

train = full_train[train_id, :]
valid = full_train[valid_id, :]

# Data cleaning
for col in [:cylindree, :consommation]
    train[!, col] = replace.(train[!, col], "," => ".")
    valid[!, col] = replace.(valid[!, col], "," => ".")
    train[!, col] = safe_parse_float.(train[!, col])
    valid[!, col] = safe_parse_float.(valid[!, col])
end

# Define categorical columns
categorical_cols = [:type, :transmission]

# Collect unique levels from the training set
levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
end

train = one_hot_encode(train, categorical_cols, levels_dict)
valid = one_hot_encode(valid, categorical_cols, levels_dict)

# Define numeric columns
numeric_cols = setdiff(names(train)[eltype.(eachcol(train)) .<: Real], [:consommation])

# Extract numeric columns
X_train = Matrix(select(train, numeric_cols))
y_train = Vector(train.consommation)
X_valid = Matrix(select(valid, numeric_cols))

# Standardize the data
X_mean = mean(X_train; dims=1)
X_stddev = std(X_train; dims=1, corrected=false)
X_train_std = (X_train .- X_mean) ./ X_stddev
X_valid_std = (X_valid .- X_mean) ./ X_stddev

# Perform PCA
pca_model = fit(PCA, X_train_std'; maxoutdim=5)
Z_train = MultivariateStats.transform(pca_model, X_train_std')'
Z_valid = MultivariateStats.transform(pca_model, X_valid_std')'

# Add principal components to DataFrames (train and valid)
for i in 1:size(Z_train, 2)
    train[:, "PC$(i)"] = Z_train[:, i]
    valid[:, "PC$(i)"] = Z_valid[:, i]
end

# Fit a regression model using PCA
model_with_pca = lm(@formula(consommation ~ PC1 + PC2 + PC3 + PC4 + PC5), train)

# Predict on validation data
valid_prediction_with_pca = predict(model_with_pca, valid)

# Calculate RMSE
rmse_with_pca = sqrt(mean((valid_prediction_with_pca - valid.consommation).^2))
println("RMSE with PCA: ", rmse_with_pca)

#Testing set
# Data cleaning for the test set

for col in [:cylindree]
    test[!, col] = replace.(test[!, col], "," => ".")
    test[!, col] = safe_parse_float.(test[!, col])
end

# One-hot encoding for the test set using levels from the training set
test = one_hot_encode(test, categorical_cols, levels_dict)

# Extract numeric columns
numeric_cols_test = names(test)[eltype.(eachcol(test)) .<: Real]
X_test = Matrix(select(test, numeric_cols_test))

# Standardize the test data using training mean and stddev
X_test_std = (X_test .- X_mean) ./ X_stddev

# Transform test data using the PCA model
Z_test = MultivariateStats.transform(pca_model, X_test_std')'

# Add principal components to the test DataFrame
for i in 1:size(Z_test, 2)
    test[:, "PC$(i)"] = Z_test[:, i]
end

# Predict 'consommation' on the test set
test_prediction_with_pca = predict(model_with_pca, test)

# Add predictions to the test DataFrame
test[:, :consommation_predite] = test_prediction_with_pca


