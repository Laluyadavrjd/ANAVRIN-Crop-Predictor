import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load and prepare data
    df = pd.read_csv('data/Crop_recommendation.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    # Save feature column order so inference can construct inputs in the same order
    # This prevents errors and silent mis-ordering between training and serving
    joblib.dump(list(X.columns), 'models/feature_order.joblib')

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model with basic parameters for maximum compatibility
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for faster training
        max_depth=10,     # Limited depth for better compatibility
        random_state=42,
        n_jobs=1          # Single job for compatibility
    )
    model.fit(X_train_scaled, y_train)

    # Calculate and print accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.2%}")

    # Save predictions for top 3 crops for each test instance
    probabilities = model.predict_proba(X_test_scaled)
    classes = model.classes_

    # Save model using joblib
    joblib.dump(model, 'models/crop_model.joblib')
    
    # Save scaler using joblib
    joblib.dump(scaler, 'models/scaler.joblib')

    # Calculate and print accuracy
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_model()
