using prediciton_consommation_carburant

train, test = load_data("data/raw/train.csv", "data/raw/test.csv")

train_processed, test_processed = preprocess_data(train, test)
save_data(train_processed, "data/processed/train_processed.csv")
save_data(test_processed, "data/processed/test_processed.csv")

model = train_model(train_processed)

predictions = predict(model, test_processed)

save_submission(predictions, "submissions/submission_2024-11-30.csv")