import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import paths
from feature_extract import extract_features_batch


def load_model(model_path: str):
    """
    Returns tuple of model, scaler 
    """

    return joblib.load(model_path)


def classify(model, scaler, audio_path: str):
    in_files = [audio_path]

    features = extract_features_batch(in_files).flatten()

    features = np.array([ features ])
    
    features = scaler.transform(features)

    return model.predict(features)[0]


def svm_fit_instrument( instrument: str,
                        param_grid: dict,
                        X_train, y_train,
                        X_val, y_val,
                        X_test, y_test ):
    NUM_CPUS=-1

    def gridSearch(estimator, para_grid, ps, X_train, y_train):
        grid = GridSearchCV( estimator=estimator,
                             param_grid=para_grid,
                             cv=ps,
                             n_jobs=NUM_CPUS,
                             verbose=2 )
        grid.fit(X_train, y_train)

        return grid.best_params_, grid.best_score_


    # Normalize features - important for SVM
    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_val_scaled   = scaler.transform(X_val)


    # Create a predefined split cross-validator to use in the grid search
    X_train_combined = np.concatenate( (X_train, X_val) )
    y_train_combined = np.concatenate( (y_train, y_val) )

    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    ps = PredefinedSplit(test_fold)

    best_params, _ = gridSearch(SVC(), param_grid, ps, X_train_combined, y_train_combined)

    svm_classifier = SVC(**best_params)
    svm_classifier.fit(X_train_scaled, y_train)


    # Save the model
    model_filename = os.path.join(paths.SVM_MODELS_DIR, f"{instrument}_model.pkl")
    joblib.dump( (svm_classifier, scaler), model_filename)

    # Test model
    y_test_pred = svm_classifier.predict(X_test_scaled)
    y_val_pred  = svm_classifier.predict(X_val_scaled)

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_accuracy  = accuracy_score(y_val,  y_val_pred)

    print()
    print(f"Best params for {instrument}:        ".ljust(45), best_params)
    print(f"Validation accuracy for {instrument}:".ljust(45), f"{val_accuracy}")
    print(f"Test accuracy for {instrument}:      ".ljust(45), f"{test_accuracy}\n")
