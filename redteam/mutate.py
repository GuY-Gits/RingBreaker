import pandas as pd
import random

def mutate_ring(ring_df: pd.DataFrame, model) -> pd.DataFrame:
    """
    F20: Red-team agent.
    Mutates ring parameters (amounts, delays) to evade the detector.
    Returns the mutated variants if the detector misses them.
    """
    if ring_df.empty:
        return ring_df

    mutated = ring_df.copy()
    
    # Mutate amounts by applying a random scaling factor (fragmenting or inflating)
    mutated['amount'] = mutated['amount'].apply(lambda x: x * random.uniform(0.6, 1.4))
    
    # Mutate velocity by adding random delays (in seconds) between hops
    if 'delay_seconds' in mutated.columns:
        mutated['delay_seconds'] = mutated['delay_seconds'].apply(
            lambda x: x + random.randint(30, 1200)
        )
    
    # Test the mutated ring against the current model
    drop_cols = ['is_fraud', 'payment_id', 'timestamp', 'ring_id']
    X_test = mutated.drop(columns=[col for col in drop_cols if col in mutated.columns])
    
    predictions = model.predict(X_test)
    
    # If the model misses more than 50% of the ring, keep this variant for retraining
    if sum(predictions) < (len(predictions) * 0.5): 
        return mutated
    
    # Otherwise, the model caught it, so discard the mutation
    return None