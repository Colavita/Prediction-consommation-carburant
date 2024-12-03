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

# ## convert annee column into age
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

# 1 if boite is automatique, 0 if manuelle
train.boite = replace(train.boite, "automatique" => 1)
train.boite = replace(train.boite, "manuelle" => 0)
valid.boite = replace(valid.boite, "automatique" => 1)
valid.boite = replace(valid.boite, "manuelle" => 0)
test.boite = replace(test.boite, "automatique" => 1)
test.boite  = replace(test.boite, "manuelle" => 0)

# change type break_petit to voiture_minicompacte
train.type = replace(train.type, "break_petit" => "voiture_minicompacte")
valid.type = replace(valid.type, "break_petit" => "voiture_minicompacte")
test.type = replace(test.type, "break_petit" => "voiture_minicompacte")

# Define categorical columns
categorical_cols = [:type, :transmission]

# Collect unique levels from the training set
levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
end

train = one_hot_encode(train, categorical_cols, levels_dict)
valid = one_hot_encode(valid, categorical_cols, levels_dict)
test = one_hot_encode(test, categorical_cols, levels_dict)

# Define the target variable
target = :consommation

# Define the feature matrix and target vector
X_train = Matrix(train[:, Not([target])])
y_train = train[!, target]
X_valid = Matrix(valid[:, Not([target])])
y_valid = valid[!, target]
X_test = Matrix(test)

# Define the model
model = lm(@formula(consommation ~ annee + boite + cylindree), train)

# Make predictions
ŷ_train = GLM.predict(model, train)
ŷ_valid = GLM.predict(model, valid)

# Compute the RMSE
rmse_train = sqrt(mean((ŷ_train .- train.consommation).^2))
rmse_valid = sqrt(mean((ŷ_valid .- valid.consommation).^2))

println("RMSE on the training set: $rmse_train")
println("RMSE on the validation set: $rmse_valid")

# Make predictions on the test set
ŷ_test = GLM.predict(model, test)

#  Prepare submission DataFrame
n_test = size(ŷ_test, 1)
id = 1:n_test
df_pred = DataFrame(id=id, consommation=ŷ_test)

# Save the predictions to a CSV file
name = string(rmse_valid) * ".csv"
CSV.write("./submissions/linear/" * name, df_pred)
println("Predictions exported successfully to " * name * ".")
