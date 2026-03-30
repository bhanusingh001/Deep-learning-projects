import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import pickle
import os

def main():
    print("Loading dataset...")
    # Load dataset
    data_path = 'breast cancer dataset.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}!")
        return

    df = pd.read_csv(data_path)
    
    # Preprocessing
    print("Preprocessing data...")
    # Drop id column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Check for empty Unnamed columns
    unnamed_cols = [col for col in df.columns if "Unnamed" in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Encode diagnosis
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Split features and target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build Neural Network
    print("Building and training Model...")
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2), # Dropout to prevent overfitting
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train Model
    history = model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), batch_size=16, verbose=1)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model trained! Test Accuracy: {accuracy*100:.2f}%")
    
    # Save Model and Scaler
    print("Saving artifacts...")
    model.save('breast_cancer_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training complete! Model saved as 'breast_cancer_model.h5' and Scaler as 'scaler.pkl'.")

if __name__ == '__main__':
    main()
