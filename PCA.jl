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

#drop boite colonnes
train = select!(train, Not(:boite))
valid = select!(valid, Not(:boite))
test = select!(test, Not(:boite))

# Define categorical columns
categorical_cols = [:type, :transmission]

# Collect unique levels from the training set
levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
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
#Remove outliers in the training set
train = remove_outliers_by_iqr(train, :cylindree, :consommation)
valid = remove_outliers_by_iqr(valid, :cylindree, :consommation)

train = one_hot_encode(train, categorical_cols, levels_dict)
valid = one_hot_encode(valid, categorical_cols, levels_dict)
test = one_hot_encode(test, categorical_cols, levels_dict)

y_train = train.consommation
X_train = select(train, Not(:consommation))
y_valid = valid.consommation
X_valid = select(valid, Not(:consommation))
X_test = deepcopy(test)

X_train = Matrix(select(train, Not(:consommation)))
X_valid = Matrix(select(valid, Not(:consommation)))
X_test = Matrix(test)

y_train = Vector(y_train)
y_valid = Vector(y_valid)

# Perform PCA
pca_model = fit(PCA, X_train'; maxoutdim=5)
Z_train = MultivariateStats.transform(pca_model, X_train')'
Z_valid = MultivariateStats.transform(pca_model, X_valid')'
Z_test = MultivariateStats.transform(pca_model, X_test')'

# Add principal components to DataFrames (train and valid)
for i in 1:size(Z_train, 2)
    train[:, "PC$(i)"] = Z_train[:, i]
    valid[:, "PC$(i)"] = Z_valid[:, i]
    test[:, "PC$(i)"] = Z_test[:, i]
end

# Fit a regression model using PCA
model_with_pca = lm(@formula(consommation ~ PC1 + PC2 + PC3 + PC4 + PC5), train)

# Predict on validation data
valid_prediction_with_pca = predict(model_with_pca, valid)

# Calculate RMSE
rmse_with_pca = sqrt(mean((valid_prediction_with_pca - valid.consommation).^2))
println("RMSE with PCA: ", rmse_with_pca)

# Prepare submission DataFrame

# Make predictions on the test set
ŷ_test = GLM.predict(model_with_pca, test)

#  Prepare submission DataFrame
n_test = size(ŷ_test, 1)
id = 1:n_test
df_pred = DataFrame(id=id, consommation=ŷ_test)

# Save the predictions to a CSV file
name = string(rmse_with_pca) * ".csv"
CSV.write("./submissions/pca/" * name, df_pred)
println("Predictions exported successfully to " * name * ".")
