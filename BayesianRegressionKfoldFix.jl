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

# # ## convert annee column into age
# train.age = 2024 .- train.annee
# valid.age = 2024 .- valid.annee
# test.age = 2024 .- test.annee

# train = select!(train, Not(:annee))
# valid = select!(valid, Not(:annee))
# test = select!(test, Not(:annee))

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

# Encode 'boite' column in all datasets
for df in [train, valid, test]
    df.boite = ifelse.(df.boite .== "automatique", 1.0, 0.0)
end

function compute_bic(X, y)
    # Ajuster un modèle
    beta = (X' * X) \ (X' * y)
    residuals = y - X * beta
    ssr = sum(residuals .^ 2) # Résidu de somme des carrés
    n = size(X, 1) # Nombre d'échantillons
    p = size(X, 2) # Nombre de variables explicatives
    bic = n * log(ssr / n) + p * log(n) # Calcul du BIC
    return bic
end

function gibbs_sampling(X, y, N)
    p = size(X, 2) # Nombre total de variables explicatives
    γ = ones(Int, p) # Initialisation : inclure toutes les variables
    best_model = deepcopy(γ)
    best_bic = Inf

    for t in 1:N
        for i in 1:p
            γ_temp = deepcopy(γ)

            # Option 1: Inclure la variable
            γ_temp[i] = 1
            X_included = X[:, γ_temp .== 1]
            bic_included = compute_bic(X_included, y)

            # Option 2: Exclure la variable
            γ_temp[i] = 0
            X_excluded = X[:, γ_temp .== 1]
            bic_excluded = compute_bic(X_excluded, y)

            # Calculer θ_i
            θ_i = exp(-bic_included) / (exp(-bic_included) + exp(-bic_excluded))

            # Tirer γ_i de Bernoulli(θ_i)
            γ[i] = rand() < θ_i ? 1 : 0
        end

        # Calculer le BIC pour le modèle courant
        X_current = X[:, γ .== 1]
        bic_current = compute_bic(X_current, y)

        # Mettre à jour le meilleur modèle
        if bic_current < best_bic
            best_bic = bic_current
            best_model = deepcopy(γ)
        end
    end

    return best_model, best_bic
end


function remove_outliers_by_iqr(df, group_col, value_col)
    return combine(groupby(df, group_col)) do sdf
        q1 = quantile(sdf[!, value_col], 0.25)
        q3 = quantile(sdf[!, value_col], 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filter(row -> lower_bound ≤ row[value_col] ≤ upper_bound, sdf)
    end
end

train = remove_outliers_by_iqr(train, :cylindree, :consommation)
valid = remove_outliers_by_iqr(valid, :cylindree, :consommation)

categorical_cols = [:type, :transmission]

levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
end

train = one_hot_encode(train, categorical_cols, levels_dict)
valid = one_hot_encode(valid, categorical_cols, levels_dict)
test = one_hot_encode(test, categorical_cols, levels_dict)





X_train = select(train, Not(:consommation))
X_valid = select(valid, Not(:consommation))
X_test = deepcopy(test)

# Identify numeric feature indices
feature_names = names(train)
numeric_features = [ :cylindree, :nombre_cylindres, :annee]
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

y_train = Vector(train.consommation)
y_valid = Vector(valid.consommation)

# Ridge regression with cross-validation
XtX = X_train' * X_train
Xty = X_train' * y_train
n_features = size(X_train, 2)

lambda_values = 10 .^ range(-5, stop=5, length=1000)
best_rmse = Inf
best_lambda = 0.0
best_beta = nothing

# for λ in lambda_values
#     local beta = (XtX + λ * I) \ Xty  # Ensure beta is local
#     y_pred_valid = X_valid * beta
#     rmse = sqrt(mean((y_pred_valid - y_valid).^2))
#     global best_rmse, best_lambda, best_beta
#     if rmse < best_rmse
#         best_rmse = rmse
#         best_lambda = λ
#         best_beta = beta
#     end
# end

# println("Best Lambda: ", best_lambda)
# println("Best RMSE: ", best_rmse)

# y_valid_pred = X_valid * best_beta
# rmse_valid = sqrt(mean((y_valid_pred - y_valid).^2))
# println("Validation RMSE: ", rmse_valid)


X_full = vcat(X_train, X_valid)
y_full = vcat(y_train, y_valid)
n_folds = 5
n_samples = size(X_full, 1)
fold_size = n_samples ÷ n_folds
folds = repeat(1:n_folds, inner=fold_size)
shuffle!(folds)
rmse_values = []
for λ in lambda_values
    rmse = 0.0
    for fold in 1:n_folds
        train_indices = findall(x -> x != fold, folds)
        valid_indices = findall(x -> x == fold, folds)
        X_train_fold = X_full[train_indices, :]
        y_train_fold = y_full[train_indices]
        X_valid_fold = X_full[valid_indices, :]
        y_valid_fold = y_full[valid_indices]

         # Compute beta using ridge regression formula
        beta = (X_train_fold' * X_train_fold + λ * I) \ (X_train_fold' * y_train_fold)
        
        # Predict on the validation fold
        y_pred_valid = X_valid_fold * beta
        
        # Calculate RMSE for the validation fold
        rmse += sqrt(mean((y_pred_valid - y_valid_fold).^2))
    end
    global best_rmse, best_lambda, best_beta
     # Average RMSE over all folds
    avg_rmse = rmse / n_folds
    push!(rmse_values, avg_rmse)
    
    # Update the best model if the current RMSE is lower
    if avg_rmse < best_rmse
        best_rmse = avg_rmse
        best_lambda = λ
        best_beta = beta
    end
end

# Plot RMSE values
Gadfly.plot(x=lambda_values, y=rmse_values, Geom.line, Guide.xlabel("λ"), Guide.ylabel("RMSE"))

# Find best λ
best_lambda = lambda_values[argmin(rmse_values)]
println("Best λ: ", best_lambda)

# # Train model with best λ
# beta = (X_train' * X_train + best_lambda * I) \ (X_train' * y_train)

# Evaluate on validation set
y_valid_pred = X_valid * beta
y_train_pred = X_train * beta
rmse_valid = sqrt(mean((y_valid_pred - y_valid).^2))
rmse_train = sqrt(mean((y_train_pred - y_train).^2))
println("Validation RMSE: ", rmse_valid)
println("Train RMSE: ", rmse_train)

best_model = gibbs_sampling(X_train, y_train, 1000)
println(best_model)

# Make prediction on validation set with best model
X_valid_best = X_valid[:, best_model[1] .== 1]
X_train_best = X_train[:, best_model[1] .== 1]

beta_best_valid = (X_valid_best' * X_valid_best) \ (X_valid_best' * y_valid)
beta_best_train = (X_train_best' * X_train_best) \ (X_train_best' * y_train)

y_valid_pred_best = exp.(X_valid_best * beta_best_valid)
y_train_pred_best = exp.(X_train_best * beta_best_train)
rmse_valid_best = sqrt(mean((y_valid_pred_best - valid.consommation).^2))
rmse_train_best = sqrt(mean((y_train_pred_best - train.consommation).^2))
println("Validation RMSE with best model: ", rmse_valid_best)
println("Train RMSE with best model: ", rmse_train_best)

# # Make prediction on test set with best model
# X_test_best = X_test[:, best_model[1] .== 1]
# y_test_pred_best = exp.(X_test_best * beta_best)

# #  Prepare submission DataFrame
# n_test = size(y_test_pred_best, 1)
# id = 1:n_test
# df_pred = DataFrame(id=id, consommation=y_test_pred_best)

# # Save the predictions to a CSV file
# name = string(rmse_valid_best) * '1' * ".csv"
# CSV.write("./submissions/bayes/" * name, df_pred)
# println("Predictions exported successfully to " * name * ".")


