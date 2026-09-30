import pandas as pd
import xgboost as xgb

def retrain_model(historical_features_path: str, confirmed_labels_path: str, model_save_path: str):
    """
    F19: Retrain on verified labels.
    Rebuilds features and retrains the model with confirmed outcomes from analysts.
    """
    try:
        # Load the original historical features and the new verified labels
        df = pd.read_csv(historical_features_path)
        labels = pd.read_csv(confirmed_labels_path)
        
        # Merge verified outcomes into the dataset based on payment_id
        df = df.merge(labels, on='payment_id', how='left')
        
        # NOTE: Once Person 2 finishes their feature scripts, you would import 
        # and run them here to rebuild the feature matrix.
        
        # Separate features (X) and target (y)
        drop_cols = ['is_fraud', 'payment_id', 'timestamp', 'ring_id']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['is_fraud']
        
        # Retrain the XGBoost pair risk model
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X, y)
        
        # Save the newly trained model
        model.save_model(model_save_path)
        
        return {
            "status": "success", 
            "accuracy": model.score(X, y),
            "message": "Model retrained successfully with verified labels."
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}