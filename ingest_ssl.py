import os
import json
import yaml
import torch
import numpy as np
import torch.nn as nn
from supabase import create_client


# --- LOAD THE SSL ENCODER ---
class SSL_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 29 features -> 384-d latent space
        self.encoder = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 384) # Our final embedding size for Supabase
        )
        # Decoder: Reconstruct back to 29 features
        self.decoder = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 29)
        )
    def forward(self, x): return self.encoder(x)



# --- CREATE A USER PROFILE (THE ANCHOR) ---
def create_user_profile(user_id, name, supabase, model):
    # Simulate an "Average" transaction for this user (29 features)
    # In a real app, this would be the mean of their last 50 transactions
    avg_txn = torch.randn(1, 29) 
    
    with torch.no_grad():
        embedding = model(avg_txn).numpy().tolist()[0]

    data = {
        "user_id": user_id,
        "full_name": name,
        "behavioral_embedding": embedding
    }
    
    supabase.table("user_profiles").upsert(data).execute()
    print(f"Profile created for {name}")

# --- THE FRAUD GUARD (THE COMPARISON) ---
def check_transaction(user_id, incoming_txn_features, supabase, model):
    # Fetch User's Anchor from Supabase
    user_data = supabase.table("user_profiles").select("behavioral_embedding").eq("user_id", user_id).execute()
    
    # Check if we actually got data
    if not user_data.data:
        print("\nUser profile not found!")
        return
    
    raw_embedding = user_data.data[0]['behavioral_embedding']

    if isinstance(raw_embedding, str):
        anchor_vec = np.array(json.loads(raw_embedding))
    else:
        anchor_vec = np.array(raw_embedding)

    # Embed the incoming transaction
    with torch.no_grad():
        new_vec_raw = model(torch.FloatTensor(incoming_txn_features).view(1, -1)).numpy()[0]
        new_vec = np.array(new_vec_raw)

    # Calculate Euclidean Distance (The 'Force Field')
    distance = np.linalg.norm(anchor_vec - new_vec)
    
    # Get Dynamic Threshold (HF part)
    config = supabase.table("system_config").select("value").eq("key", "anomaly_threshold").execute()
    threshold = config.data[0]['value']

    is_anomaly = bool(distance > threshold)
    
    print(f"Distance: {distance:.4f} | Threshold: {threshold} | Anomaly: {is_anomaly}")
    
    # Store the transaction
    supabase.table("transactions").insert({
                                            "user_id": user_id,
                                            "amount": round(np.random.uniform(10, 5000), 2),       # You can pull this from the 'Amount' col in Kaggle
                                            "merchant": np.random.choice(["Amazon", "Apple", "Uber", "Unknown_Proxy"]),  # Simulated for the UI
                                            "location": "Toronto, ON", # Simulated for the UI
                                            "embedding": new_vec.tolist(),
                                            "is_anomaly": is_anomaly
                                        }).execute()

    return is_anomaly


def update_threshold_with_feedback(is_correct_catch, supabase):
    # Get current threshold
    res = supabase.table("system_config").select("value").eq("key", "anomaly_threshold").execute()
    current_val = res.data[0]['value']
    
    # If the AI missed a fraud (False Negative), lower the threshold (be more sensitive)
    # If the AI flagged a normal txn (False Positive), raise the threshold (be more relaxed)
    if is_correct_catch:
        new_val = current_val # Keep it as is
    else:
        # If we had a False Positive, we increase threshold by 5%
        new_val = current_val * 1.05 
        
    supabase.table("system_config").update({"value": new_val}).eq("key", "anomaly_threshold").execute()
    print(f"RLHF Update: New Threshold is {new_val:.4f}")

def main(config):

    # --- CONFIGURATION ---
    SUPABASE_URL = config["supabase_url"]
    SUPABASE_KEY = config["supabase_key"] # Use Service Role to bypass RLS
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    model = SSL_Encoder()
    model.load_state_dict(torch.load(config["model_path"], map_location=torch.device('cpu')))
    model.eval()
    
    
    # --- TEST THE SYSTEM ---
    # Setup a user
    create_user_profile("user_123", "John Doe", supabase, model)

    # Simulate a "Normal" transaction (Low distance)
    check_transaction("user_123", torch.randn(29).tolist(), supabase, model)

    # Simulate an "Outlier" (High distance - multiple by 5 to force anomaly)
    check_transaction("user_123", (torch.randn(29) * 5.0).tolist(), supabase, model)


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_config = os.path.join(ROOT_DIR, "config.yaml")
    
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)