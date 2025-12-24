
# Fit model and Cross-Validation, ARCHITECTURE 1 SIMPLE LSTM
epochs = 3
batch_size = 32

model = simple_lstm()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)

print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
print_layers_dims(model)

