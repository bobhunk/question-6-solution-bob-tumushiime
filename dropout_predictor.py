import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

class DropoutRiskPredictor:
    """
    Comprehensive machine learning system for predicting school dropout risks
    in Sub-Saharan African countries with focus on actionable insights.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.target_column = 'dropout_risk_high'
        self.model_performance = {}
        
    def load_and_prepare_data(self, data_path=None):
        """
        Load and prepare data for machine learning modeling
        
        Args:
            data_path (str): Path to the dataset file
            
        Returns:
            tuple: X (features), y (target), feature_names
        """
        if data_path is None:
            # Try to find the most recent processed data
            import glob
            feature_files = glob.glob('data/processed/dropout_risk_features_*.csv')
            if feature_files:
                data_path = sorted(feature_files)[-1]
            else:
                raise FileNotFoundError("No processed data found. Run data_collection.py first.")
        
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        print(f"Original dataset shape: {df.shape}")
        
        # Feature selection for modeling
        feature_columns = [
            # Enrollment rates
            'SE.PRM.NENR', 'SE.SEC.NENR', 'SE.PRM.NENR.FE', 'SE.SEC.NENR.FE',
            'SE.PRM.NENR.MA', 'SE.SEC.NENR.MA',
            
            # Completion rates
            'SE.PRM.CMPT.ZS', 'SE.PRM.CMPT.FE.ZS', 'SE.PRM.CMPT.MA.ZS',
            'SE.SEC.CMPT.LO.ZS', 'SE.SEC.CMPT.LO.FE.ZS', 'SE.SEC.CMPT.LO.MA.ZS',
            
            # Persistence and survival
            'SE.PRM.PRSL.ZS', 'SE.PRM.PRSL.FE.ZS', 'SE.PRM.PRSL.MA.ZS',
            
            # Economic indicators
            'SE.XPD.TOTL.GD.ZS', 'NY.GDP.PCAP.CD', 'SP.RUR.TOTL.ZS',
            
            # Literacy and demographics
            'SE.ADT.LITR.ZS', 'SE.ADT.LITR.FE.ZS', 'SE.ADT.LITR.MA.ZS',
            'SP.DYN.TFRT.IN',
            
            # Engineered features
            'primary_gender_parity', 'secondary_gender_parity',
            'primary_to_secondary_transition', 'primary_completion_gap'
        ]
        
        # Add categorical features
        categorical_features = ['country_code', 'region']
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df.columns]
        available_categorical = [col for col in categorical_features if col in df.columns]
        
        print(f"Available numerical features: {len(available_features)}")
        print(f"Available categorical features: {len(available_categorical)}")
        
        # Prepare feature matrix
        X_numerical = df[available_features].copy()
        X_categorical = df[available_categorical].copy()
        
        # Encode categorical variables
        label_encoders = {}
        for col in available_categorical:
            le = LabelEncoder()
            X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))
            label_encoders[col] = le
        
        # Combine features
        X = pd.concat([X_numerical, X_categorical], axis=1)
        
        # Target variable
        if self.target_column in df.columns:
            y = df[self.target_column].copy()
        else:
            print(f"Target column '{self.target_column}' not found. Creating based on completion rates...")
            # Create target based on primary completion rate
            if 'SE.PRM.CMPT.ZS' in df.columns:
                y = (df['SE.PRM.CMPT.ZS'] < 70).astype(int)  # High risk if completion < 70%
            else:
                raise ValueError("Cannot create target variable. Insufficient data.")
        
        # Handle missing values
        print("Handling missing values...")
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Remove rows with too many missing values in original data
        missing_threshold = 0.7  # Keep rows with at least 70% non-missing values
        valid_rows = X.notna().sum(axis=1) >= (len(X.columns) * missing_threshold)
        
        X_final = X_imputed[valid_rows]
        y_final = y[valid_rows]
        
        print(f"Final dataset shape: {X_final.shape}")
        print(f"Target distribution: {y_final.value_counts().to_dict()}")
        
        self.feature_names = list(X_final.columns)
        self.label_encoders = label_encoders
        self.imputer = imputer
        
        return X_final, y_final, self.feature_names
    
    def train_models(self, X, y):
        """
        Train multiple machine learning models for dropout prediction
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
        """
        print("Training multiple machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        # Model configurations
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'use_scaled': False
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'use_scaled': True
            }
        }
        
        # Train and evaluate models
        for model_name, config in models_config.items():
            print(f"\nTraining {model_name}...")
            
            # Select appropriate data
            X_train_model = X_train_scaled if config['use_scaled'] else X_train
            X_test_model = X_test_scaled if config['use_scaled'] else X_test
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_model, y_train)
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_model)
            y_pred_proba = best_model.predict_proba(X_test_model)[:, 1]
            
            # Evaluation metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                best_model, X_train_model, y_train, 
                cv=5, scoring='roc_auc'
            )
            
            # Store results
            self.models[model_name] = best_model
            self.model_performance[model_name] = {
                'best_params': grid_search.best_params_,
                'test_auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'use_scaled': config['use_scaled']
            }
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Test AUC: {auc_score:.3f}")
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    def evaluate_models(self):
        """
        Evaluation of trained models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Performance comparison
        performance_df = pd.DataFrame({
            model_name: {
                'Test AUC': perf['test_auc'],
                'CV Mean AUC': perf['cv_mean'],
                'CV Std AUC': perf['cv_std']
            }
            for model_name, perf in self.model_performance.items()
        }).T
        
        print(performance_df.round(3))
        
        # Best model
        best_model_name = performance_df['Test AUC'].idxmax()
        print(f"\nBest performing model: {best_model_name}")
        
        # Detailed evaluation for best model
        best_perf = self.model_performance[best_model_name]
        
        print(f"\n{best_model_name} - Detailed Results:")
        print(f"Best parameters: {best_perf['best_params']}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, best_perf['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_perf['predictions'])
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        performance_df['Test AUC'].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Model Performance Comparison (Test AUC)')
        axes[0,0].set_ylabel('AUC Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,1], cmap='Blues')
        axes[0,1].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, best_perf['probabilities'])
        axes[1,0].plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_perf["test_auc"]:.3f})')
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,0].set_xlabel('False Positive Rate')
        axes[1,0].set_ylabel('True Positive Rate')
        axes[1,0].set_title('ROC Curve')
        axes[1,0].legend()
        
        # 4. Feature importance (if available)
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models[best_model_name].feature_importances_
            }).sort_values('importance', ascending=True)
            
            importance_df.tail(10).plot(x='feature', y='importance', kind='barh', ax=axes[1,1])
            axes[1,1].set_title('Top 10 Feature Importance')
            axes[1,1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name
    
    def predict_dropout_risk(self, country_data, model_name=None):
        """
        Predict dropout risk for new data
        
        Args:
            country_data (dict or pandas.DataFrame): Country/district data
            model_name (str): Model to use for prediction
            
        Returns:
            dict: Prediction results with probabilities and risk level
        """
        if model_name is None:
            # Use best performing model
            performance_df = pd.DataFrame({
                model_name: {'Test AUC': perf['test_auc']}
                for model_name, perf in self.model_performance.items()
            }).T
            model_name = performance_df['Test AUC'].idxmax()
        
        model = self.models[model_name]
        use_scaled = self.model_performance[model_name]['use_scaled']
        
        # Prepare input data
        if isinstance(country_data, dict):
            input_df = pd.DataFrame([country_data])
        else:
            input_df = country_data.copy()
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = np.nan
        
        # Select and order features
        input_features = input_df[self.feature_names]
        
        # Handle missing values
        input_imputed = pd.DataFrame(
            self.imputer.transform(input_features),
            columns=self.feature_names
        )
        
        # Scale if needed
        if use_scaled:
            input_scaled = self.scalers['standard'].transform(input_imputed)
            prediction_input = input_scaled
        else:
            prediction_input = input_imputed
        
        # Make predictions
        risk_probability = model.predict_proba(prediction_input)[:, 1]
        risk_prediction = model.predict(prediction_input)
        
        # Risk categorization
        risk_levels = []
        for prob in risk_probability:
            if prob < 0.3:
                risk_levels.append('Low Risk')
            elif prob < 0.7:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('High Risk')
        
        return {
            'model_used': model_name,
            'risk_probability': risk_probability,
            'risk_prediction': risk_prediction,
            'risk_level': risk_levels
        }
    
    def save_models(self):
        """
        Save trained models and preprocessing objects
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            filename = f"models/{model_name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save preprocessing objects
        joblib.dump(self.scalers, 'models/scalers.joblib')
        joblib.dump(self.imputer, 'models/imputer.joblib')
        joblib.dump(self.feature_names, 'models/feature_names.joblib')
        
        # Save performance metrics
        performance_df = pd.DataFrame(self.model_performance).T
        performance_df.to_csv('models/model_performance.csv')
        
        print("All models and preprocessing objects saved successfully!")

def main():
    """
    Main execution function for dropout risk prediction
    """
    print("="*80)
    print("AFRICAN SCHOOL DROPOUT RISK PREDICTION")
    print("="*80)
    
    # Initialize predictor
    predictor = DropoutRiskPredictor()
    
    try:
        # Load and prepare data
        X, y, feature_names = predictor.load_and_prepare_data()
        
        # Train models
        predictor.train_models(X, y)
        
        # Evaluate models
        best_model = predictor.evaluate_models()
        
        # Save models
        predictor.save_models()
        
        print(f"\n{'='*80}")
        print("DROPOUT RISK PREDICTION SYSTEM READY")
        print(f"{'='*80}")
        print(f"Best model: {best_model}")
        print(f"Features used: {len(feature_names)}")
        print(f"Training samples: {len(X)}")
        print("\nModel files saved in 'models/' directory")
        print("Ready for interactive story and policy recommendations!")
        
        # Example prediction
        print(f"\n{'='*40}")
        print("EXAMPLE PREDICTION")
        print(f"{'='*40}")
        
        # Create example country data
        example_data = {
            'SE.PRM.NENR': 85.0,  # Primary enrollment
            'SE.SEC.NENR': 45.0,  # Secondary enrollment  
            'SE.PRM.CMPT.ZS': 65.0,  # Primary completion
            'SE.ADT.LITR.ZS': 70.0,  # Adult literacy
            'NY.GDP.PCAP.CD': 800.0,  # GDP per capita
            'SP.RUR.TOTL.ZS': 75.0,  # Rural population
        }
        
        # Fill missing features with median values
        for feature in feature_names:
            if feature not in example_data:
                example_data[feature] = X[feature].median()
        
        result = predictor.predict_dropout_risk(example_data)
        
        print(f"Example country profile:")
        print(f"- Primary enrollment: 85%")
        print(f"- Secondary enrollment: 45%") 
        print(f"- Primary completion: 65%")
        print(f"- Adult literacy: 70%")
        print(f"- GDP per capita: $800")
        print(f"- Rural population: 75%")
        print(f"\nPrediction:")
        print(f"- Risk probability: {result['risk_probability'][0]:.1%}")
        print(f"- Risk level: {result['risk_level'][0]}")
        print(f"- Model used: {result['model_used']}")
        
    except Exception as e:
        print(f"Error in model training: {e}")
        print("Please ensure data collection has been completed successfully.")

if __name__ == "__main__":
    main()
