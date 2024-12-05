using CSV, DataFrames, Statistics, Dates, Gadfly, Combinatorics, Plots, StatsBase, StatsPlots, Random, StatsModels, GLM, LinearAlgebra, MultivariateStats, Distributions

full_train = CSV.read("./data/raw/train.csv", DataFrame; delim=";")
test =  CSV.read("./data/raw/test.csv", DataFrame; delim=";") #ne contient pas la varialbe consommation

Random.seed!(1234) #pour la reproductibilit

ntrain = round(Int, 1*nrow(full_train)) #80% des données pour l'entrainement: 80% * nb de lignes

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

# Encode 'boite' column in all datasets
for df in [train, valid, test]
    df.boite = ifelse.(df.boite .== "automatique", 1.0, 0.0)
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


# if transmission is 4x4, change for integrale
train.transmission = ifelse.(train.transmission .== "4x4", "integrale", train.transmission)
valid.transmission = ifelse.(valid.transmission .== "4x4", "integrale", valid.transmission)
test.transmission = ifelse.(test.transmission .== "4x4", "integrale", test.transmission)

train.transmission = ifelse.(train.transmission .== "propulsion", "integrale", train.transmission)
valid.transmission = ifelse.(valid.transmission .== "propulsion", "integrale", valid.transmission)
test.transmission = ifelse.(test.transmission .== "propulsion", "integrale", test.transmission)

function categorize_vehicle_type(vehicle_type)
    if vehicle_type in ["break_petit", "voiture_minicompacte", "voiture_compacte", "VUS_petit", "voiture_moyenne"]
        return "petits_véhicules"
    elseif vehicle_type in ["voiture_sous_compacte", "break_moyen", "monospace", "camionnette_petit"]
        return "véhicules_moyens"
    else
        return "grands_véhicules"
    end
end

# train[:, :categorie_transmission] = [categorize_transmission(t) for t in train[:, :transmission]]
# valid[:, :categorie_transmission] = [categorize_transmission(t) for t in valid[:, :transmission]]
# test[:, :categorie_transmission] = [categorize_transmission(t) for t in test[:, :transmission]]

train[:, :categorie_vehicule] = [categorize_vehicle_type(t) for t in train[:, :type]]
valid[:, :categorie_vehicule] = [categorize_vehicle_type(t) for t in valid[:, :type]]
test[:, :categorie_vehicule] = [categorize_vehicle_type(t) for t in test[:, :type]]

# train = select!(train, Not( :type))
# valid = select!(valid, Not( :type))
# test = select!(test, Not(:type))

# Define categorical columns
categorical_cols = [:categorie_vehicule, :transmission, :type]

# Collect unique levels from the training set
levels_dict = Dict()
for col in categorical_cols
    levels_dict[col] = unique(train[!, col])
end

# apply log transformation to the cylindree
train.cylindree = log.(train.cylindree)
valid.cylindree = log.(valid.cylindree)
test.cylindree = log.(test.cylindree)

train.nombre_cylindres = log.(train.nombre_cylindres)
valid.nombre_cylindres = log.(valid.nombre_cylindres)
test.nombre_cylindres = log.(test.nombre_cylindres)

# #standardize the data
# function standardize(df)
#     for col in names(df)
#         if eltype(df[!, col]) <: Number
#             df[!, col] = (df[!, col] .- mean(df[!, col])) ./ std(df[!, col])
#         end
#     end
#     return df
# end

# numeric_cols = [:age, :cylindree]

# train[!, numeric_cols] = standardize(train[!, numeric_cols])
# valid[!, numeric_cols] = standardize(valid[!, numeric_cols])
# test[!, numeric_cols] = standardize(test[!, numeric_cols])

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
# model = lm(@formula(consommation ~ age + transmission_4x4+ transmission_integrale + transmission_propulsion + transmission_traction + boite + cylindree), train)
#model = lm(@formula(consommation ~ age + categorie_transmission_AWDRWD + categorie_transmission_traction + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + boite + cylindree), train) #Meilleur
# model = lm(@formula(consommation ~ age + categorie_transmission_AWDRWD + categorie_transmission_traction + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + boite + nombre_cylindres), train)
model = lm(@formula(consommation ~ age +  transmission_integrale + transmission_traction + boite + cylindree), train)

#cross validation
data_k_folds = vcat(train, valid)
y = data_k_folds.consommation
X = select(data_k_folds, Not(:consommation))

n = nrow(data_k_folds)
k = 10
fold_size = n ÷ k

indices = randperm(n)

rms_scores = []

for i in 0:(k-1)
    valid_indices = (i * fold_size + 1):((i + 1) * fold_size)
    train_indices = setdiff(1:n, valid_indices)
    
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_valid = X[valid_indices, :]
    y_valid = y[valid_indices]

    model = lm(@formula(consommation ~ age + transmission_integrale + transmission_traction + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + boite + cylindree), data_k_folds) #Meilleur
    
    ŷ_valid = GLM.predict(model, X_valid)
    rms = sqrt(mean((ŷ_valid .- y_valid).^2))
    push!(rms_scores, rms)
end

moyenne_rmse = mean(rms_scores)
println("Moyenne RMSE k-fold : $moyenne_rmse")

# Make predictions
ŷ_train = GLM.predict(model, train)
ŷ_valid = GLM.predict(model, valid)

# Compute the RMSE
rmse_train = sqrt(mean((ŷ_train .- train.consommation).^2))
rmse_valid = sqrt(mean((ŷ_valid .- valid.consommation).^2))

#plot dirstribtuion of residuals
residuals = ŷ_valid .- valid.consommation
histogram(residuals, bins=50, title="Distribution of residuals", xlabel="Residuals", ylabel="Frequency")


println("RMSE on the training set: $rmse_train")
println("RMSE on the validation set: $rmse_valid")

##############################################################################
# Define the feature matrix and target vector
X_train = Matrix(train[:, Not([:consommation])])
y_train = Vector(train.consommation)
X_valid = Matrix(valid[:, Not([:consommation])])
y_valid = Vector(valid.consommation)
X_test = Matrix(test)

function compute_raj(X, y, ŷ)
    println("Computing R^2 adjusted...")
    SSe = sum((ŷ .- y).^2)
    SSt = sum((y .- mean(y)).^2)
    println("SSt: $SSt")
    n = size(y, 1)
    p = size(X, 2)
    println("SSe: $SSe")
    return 1 - (SSe / (n - p - 1)) / (SSt / (n - 1))
end

function compute_bic(X, model)
    # Compute log-likelihood
    log_likelihood = loglikelihood(model)

    # Number of observations and parameters
    n = size(X, 1)  # Number of observations
    k = length(coef(model))  # Number of estimated parameters

    # Compute BIC
    bic = log_likelihood - (k/2) * log(n)

    return bic # Return both BIC and the fitted model
end

models_formulas = [
    @formula(consommation ~ nombre_cylindres),
    @formula(consommation ~ cylindree),
    @formula(consommation ~ cylindree + nombre_cylindres),
    @formula(consommation ~ cylindree + boite + age),
    @formula(consommation ~ cylindree + transmission_integrale),
    @formula(consommation ~ transmission_integrale + transmission_traction),
    @formula(consommation ~ transmission_integrale + nombre_cylindres),
    @formula(consommation ~ transmission_integrale + transmission_traction + cylindree),
    @formula(consommation ~ transmission_integrale + cylindree),
    @formula(consommation ~ transmission_integrale + cylindree),
    @formula(consommation ~ transmission_integrale + cylindree + nombre_cylindres),
    @formula(consommation ~ transmission_traction + cylindree + nombre_cylindres),
    @formula(consommation ~ transmission_integrale + cylindree + nombre_cylindres + boite),
    @formula(consommation ~ transmission_traction + cylindree + nombre_cylindres + boite),
    @formula(consommation ~ age + transmission_integrale + transmission_traction + cylindree + nombre_cylindres + boite),
    @formula(consommation ~ transmission_traction + transmission_integrale + cylindree + nombre_cylindres + boite),
    @formula(consommation ~ transmission_integrale + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + boite + cylindree),
    @formula(consommation ~ transmission_integrale + transmission_traction + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + cylindree),
    @formula(consommation ~ transmission_integrale + transmission_traction + categorie_vehicule_petits_véhicules + categorie_vehicule_grands_véhicules + categorie_vehicule_véhicules_moyens + cylindree + boite),
    @formula(consommation ~ age + categorie_vehicule_petits_véhicules + categorie_vehicule_véhicules_moyens + categorie_vehicule_grands_véhicules + transmission_traction + cylindree + boite),   
    @formula(consommation ~ categorie_vehicule_petits_véhicules + categorie_vehicule_véhicules_moyens + categorie_vehicule_grands_véhicules + transmission_traction + cylindree + boite),
    @formula(consommation ~ transmission_traction  + cylindree + boite),
    @formula(consommation ~ transmission_integrale + cylindree + boite),
    @formula(consommation ~ transmission_traction + cylindree + type_voiture_moyenne + type_VUS_petit + type_voiture_compacte + type_voiture_deux_places + type_voiture_minicompacte + type_VUS_standard + type_voiture_sous_compacte + type_break_petit + type_voiture_grande + type_camionnette_standard + transmission_integrale + boite),
    @formula(consommation ~ cylindree +  type_voiture_moyenne + type_VUS_petit + type_voiture_compacte + type_voiture_deux_places + type_voiture_minicompacte + type_VUS_standard + type_voiture_sous_compacte + type_break_petit + type_voiture_grande + type_camionnette_standard + transmission_integrale + transmission_traction + boite),
    @formula(consommation ~ transmission_integrale + transmission_traction + cylindree + type_voiture_moyenne + type_VUS_petit + type_voiture_compacte + type_voiture_deux_places + type_voiture_minicompacte + type_VUS_standard + type_voiture_sous_compacte + type_break_petit + type_voiture_grande + type_camionnette_standard),
    @formula(consommation ~ age + cylindree + type_voiture_moyenne + type_VUS_petit + type_voiture_compacte + type_voiture_deux_places + type_voiture_minicompacte + type_VUS_standard + type_voiture_sous_compacte + type_break_petit + type_voiture_grande + type_camionnette_standard)
    ]

    
best_rajs= -Inf
best_rmse = Inf
best_bic = -Inf
bic_rmse = Inf
best_model_by_raj = nothing
best_model_by_rmse = nothing
best_model_by_bic = nothing

for formula in models_formulas
    model = lm(formula, train)
    println(model)
    ŷ = GLM.predict(model, valid)
    local raj_valid = compute_raj(X_valid, y_valid, ŷ)
    local bic_valid = compute_bic(X_valid, model)

    println("Raj on the training set: $raj_train")
    println("Raj on the validation set: $raj_valid")

    local rmse_valid = sqrt(mean((ŷ .- y_valid).^2))
    println("RMSE on the validation set: $rmse_valid")
    global best_rmse, best_rajs, best_model_by_rmse, best_model_by_raj, best_bic, best_model_by_bic, bic_rmse
    if rmse_valid < best_rmse
        best_rmse = rmse_valid
        best_model_by_rmse = formula
    end

    if bic_valid > best_bic
        best_bic = bic_valid
        best_model_by_bic = formula
        bic_rmse = rmse_valid
    end

    if raj_valid > best_rajs
        best_rajs = raj_valid
        best_model_by_raj = formula
    end
end

#print best model 
println("\n-------------------------------------")
println("Best model by Raj: ", best_model_by_raj)
println("Best Raj: ", best_rajs)

println("Best model by RMSE: ", best_model_by_rmse)
println("Best RMSE: ", best_rmse)

println("Best model by BIC: ", best_model_by_bic)
println("Best BIC: ", best_bic)
println("RMSE for the best BIC: ", bic_rmse)
println("-------------------------------------")

###############################################################################

# Make predictions on the test set
ŷ_test = GLM.predict(lm(best_model_by_rmse, train), test)

#  Prepare submission DataFrame
n_test = size(ŷ_test, 1)
id = 1:n_test
df_pred = DataFrame(id=id, consommation=ŷ_test)

println(names(train))

# Save the predictions to a CSV file
name = "final" * ".csv"
CSV.write("./submissions/linear/" * name, df_pred)
println("Predictions exported successfully to " * name * ".")