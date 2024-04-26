import os
import joblib
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import Sequential                          # type: ignore
from tensorflow.keras.layers import Dense, Dropout                      # type: ignore
from tensorflow.keras.regularizers import l2                            # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint   # type: ignore
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import paths
from feature_extract import extract_features_batch

MODEL = Sequential(
    [
        # assuming VGGish embeddings of size 128
        Dense(256, activation='relu', input_shape=(1280,), kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1280, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1280, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ]
)

def _probability_to_binary(probabilities: np.ndarray) -> np.ndarray:
    return (probabilities > 0.5).astype(int)

def load_model(model_path: str):
    """
    Returns tuple of model, scaler 
    """
    scaler_filename = model_path.replace('_model.keras', '_scaler.pkl')
    scaler = joblib.load(scaler_filename)

    MODEL.load_weights(model_path)

    return MODEL, scaler

def classify(model, scaler, audio_path: str):
    in_files = [audio_path]

    features = extract_features_batch(in_files).flatten()

    features = np.array([ features ])
    
    features = scaler.transform(features)

    prob = model.predict(features)[0]

    return _probability_to_binary(prob)[0], prob[0]


def mlp_fit_instrument( instrument: str,
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test ):

    MODEL.compile( optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'] )

    # Callbacks for early stopping and saving the model
    model_filename = os.path.join(paths.MLP_MODELS_DIR, f'{instrument}_model.keras')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True)
    ]

    # Convert all data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val   = np.array(X_val)
    y_val   = np.array(y_val)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    # Normalize features - important for MLP
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # Assume X_train, y_train, X_val, y_val are predefined
    history = MODEL.fit(X_train_scaled, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val_scaled, y_val),
                        callbacks=callbacks)

    # Load the best model
    MODEL.load_weights(model_filename)

    # Test model
    y_test_pred_probs = MODEL.predict(X_test_scaled)
    y_val_pred_probs  = MODEL.predict(X_val_scaled)

    # Convert probabilities to binary predictions with 0.5 threshold
    y_test_pred = _probability_to_binary(y_test_pred_probs)
    y_val_pred  = _probability_to_binary(y_val_pred_probs)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_accuracy  = accuracy_score(y_val, y_val_pred)

    print()
    print(f"Validation accuracy for {instrument}:".ljust(45), f"{val_accuracy}")
    print(f"Test accuracy for {instrument}:      ".ljust(45), f"{test_accuracy}\n")

    # Save the scaler
    scaler_filename = os.path.join(paths.MLP_MODELS_DIR, f'{instrument}_scaler.pkl')

    joblib.dump(scaler, scaler_filename)
