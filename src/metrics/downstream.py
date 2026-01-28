"""Downstream classification evaluation for PCA."""

import numpy as np
from typing import Dict, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pca_model,
    classifier: str = 'knn',
    **classifier_kwargs,
) -> Dict[str, float]:
    """Evaluate classification accuracy using PCA-projected features.

    Args:
        X_train: Training data, shape (n_train, n_features).
        y_train: Training labels, shape (n_train,).
        X_test: Test data, shape (n_test, n_features).
        y_test: Test labels, shape (n_test,).
        pca_model: Fitted PCA model for projection.
        classifier: Classifier type: 'knn', 'svm', or 'logistic'.
        **classifier_kwargs: Additional kwargs for classifier.

    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics.
    """
    # Project data
    X_train_pca = pca_model.transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    # Create classifier
    if classifier == 'knn':
        n_neighbors = classifier_kwargs.get('n_neighbors', 5)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif classifier == 'svm':
        C = classifier_kwargs.get('C', 1.0)
        kernel = classifier_kwargs.get('kernel', 'rbf')
        clf = SVC(C=C, kernel=kernel)
    elif classifier == 'logistic':
        C = classifier_kwargs.get('C', 1.0)
        max_iter = classifier_kwargs.get('max_iter', 1000)
        clf = LogisticRegression(C=C, max_iter=max_iter)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    # Train and predict
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    # Compute metrics
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }


def compare_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pca_model_test,
    pca_model_baseline,
    classifier: str = 'knn',
) -> Dict[str, float]:
    """Compare classification performance between two PCA models.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        pca_model_test: Model being evaluated.
        pca_model_baseline: Baseline model for comparison.
        classifier: Classifier type.

    Returns:
        Dictionary with metrics for both models and differences.
    """
    metrics_test = evaluate_classification(
        X_train, y_train, X_test, y_test, pca_model_test, classifier
    )
    metrics_baseline = evaluate_classification(
        X_train, y_train, X_test, y_test, pca_model_baseline, classifier
    )

    return {
        'test_accuracy': metrics_test['accuracy'],
        'baseline_accuracy': metrics_baseline['accuracy'],
        'accuracy_diff': metrics_test['accuracy'] - metrics_baseline['accuracy'],
        'test_f1': metrics_test['f1'],
        'baseline_f1': metrics_baseline['f1'],
        'f1_diff': metrics_test['f1'] - metrics_baseline['f1'],
    }
