"""
VULNERABLE HEALTHCARE ML PIPELINE - Security Issues:
1. Hardcoded credentials (database password, API keys)
2. Unsafe pickle serialization for model loading
3. No input validation or sanitization
4. Vulnerable dependencies with known CVEs
5. GPL license violations
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# VULNERABILITY 1: Hardcoded Credentials
DB_PASSWORD = "admin123"  # Bandit will flag B105
API_KEY = "sk-healthcare-abc123xyz"  # Exposed secret
DATABASE_URL = "postgresql://admin:admin123@localhost/healthdb"


class InsecureHealthcarePipeline:
    def __init__(self):
        self.model = None
        self.data = None
    
    # VULNERABILITY 2: Unsafe Pickle Loading
    def load_model(self, model_path):
        """Load model using pickle - DANGEROUS!"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)  # Bandit B301
        print("Model loaded with pickle")
    
    def save_model(self, model_path):
        """Save model using pickle"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    # VULNERABILITY 3: No Input Validation
    def load_data(self, csv_path):
        """Load data without validation"""
        self.data = pd.read_csv(csv_path)  # No path validation
        return self.data
    
    def preprocess_data(self):
        """Basic preprocessing"""
        # Create dummy target for risk level
        self.data['Risk_Level'] = self.data['Billing Amount'].apply(
            lambda x: 2 if x > 30000 else (1 if x > 15000 else 0)
        )
        
        # Select numeric features
        X = self.data[['Age', 'Billing Amount']].fillna(0)
        y = self.data['Risk_Level']
        return X, y
    
    def train_model(self):
        """Train Random Forest model"""
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train, y_train)
        
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    
    def predict_risk(self, age, billing_amount):
        """Predict patient risk level"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        prediction = self.model.predict([[age, billing_amount]])
        risk_levels = ['Low', 'Medium', 'High']
        return risk_levels[prediction[0]]


def main():
    print("=== INSECURE HEALTHCARE ML PIPELINE ===")
    print(f"Using hardcoded credentials: {DB_PASSWORD}")
    print(f"API Key: {API_KEY}\n")
    
    pipeline = InsecureHealthcarePipeline()
    
    # Load and train model
    print("Loading healthcare dataset...")
    pipeline.load_data('healthcare_dataset.csv')
    
    print("Training model...")
    pipeline.train_model()
    
    # Save with insecure pickle
    print("\nSaving model with pickle...")
    pipeline.save_model('insecure_model.pkl')
    
    print("\n=== SECURITY VULNERABILITIES PRESENT ===")
    print("1. Hardcoded credentials exposed")
    print("2. Unsafe pickle serialization")
    print("3. No input validation")
    print("4. Vulnerable dependencies")
    print("\nRun static analysis tools to identify all issues!")


if __name__ == "__main__":
    main()
