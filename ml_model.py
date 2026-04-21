# ml_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_future_kwh_with_rf(history_df, future_context):
    """
    REGRESSOR: Trains on the user's history to predict the exact numerical kWh for next month.
    """
    context_mapping = {"Normal Month": 0, "Christmas/New Year": 1, "Holy Week": 1, "Summer Break": 2, "Family Occasion": 3}
    
    # The AI needs at least 3 months of history to find a pattern. 
    if len(history_df) < 3:
        return history_df['actual_kwh'].mean()
        
    # 1. Prepare the Training Data
    X_train = history_df['context'].map(context_mapping).fillna(0).values.reshape(-1, 1)
    y_train = history_df['actual_kwh'].values
    
    # 2. Train the Random Forest Regressor
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)
    
    # 3. Predict the future kWh based on the upcoming event
    future_val = np.array([[context_mapping.get(future_context, 0)]])
    ai_predicted_kwh = regr.predict(future_val)[0]
    
    return ai_predicted_kwh