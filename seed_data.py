import os
import yaml
import uuid
import torch
import numpy as np
import torch.nn as nn
from supabase import create_client


# Load your encoder (Ensure the class matches your script)
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


# --- DATA GENERATOR HELPERS ---
def get_embedding(features, model):
    with torch.no_grad():
        return model(torch.FloatTensor(features).view(1, -1)).numpy().tolist()[0]




def main(config):

    # --- CONFIGURATION ---
    SUPABASE_URL = config["supabase_url"]
    SUPABASE_KEY = config["supabase_key"] # Use Service Role to bypass RLS
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    model = SSL_Encoder()
    model.load_state_dict(torch.load(config["model_path"], map_location=torch.device('cpu')))
    model.eval()

    # --- SCENARIO GENERATION ---
    scenarios = [
        {
            "user_id": "user_toronto_01",
            "name": "Sarah Jenkins",
            "city": "Toronto, ON",
            "behavior": "Conservative/Local",
            "base_vector": torch.randn(29) * 0.5 # Tight cluster
        },
        {
            "user_id": "user_vancouver_02",
            "name": "David Chen",
            "city": "Vancouver, BC",
            "behavior": "High-Volume Tech",
            "base_vector": torch.randn(29) * 1.2 # Broader cluster
        },
        {
            "user_id": "user_singapore_03",
            "name": "Priya Sharma",
            "city": "Singapore, SG",
            "behavior": "International Traveler",
            "base_vector": torch.randn(29) * 2.0 # Volatile cluster
        }
    ]

    print("🚀 Seeding Database...")

    for s in scenarios:
        # Create Profile
        anchor = get_embedding(s['base_vector'].tolist(), model)
        supabase.table("user_profiles").upsert({
            "user_id": s['user_id'],
            "full_name": s['name'],
            "behavioral_embedding": anchor
        }).execute()
        
        # Add 5 "Normal" Transactions (Low noise)
        for _ in range(5):
            noise = torch.randn(29) * 0.1
            txn_features = (s['base_vector'] + noise).tolist()
            emb = get_embedding(txn_features, model)
            
            supabase.table("transactions").insert({
                "user_id": s['user_id'],
                "amount": round(np.random.uniform(10, 100), 2),
                "merchant": np.random.choice(["Tim Hortons", "Starbucks", "Amazon", "Local Transit"]),
                "location": s['city'],
                "embedding": emb,
                "is_anomaly": False
            }).execute()

        # Add 1 "Anomalous" Transaction (High noise/offset)
        outlier_noise = torch.randn(29) * 15.0 # Force a high Euclidean distance
        txn_features = (s['base_vector'] + outlier_noise).tolist()
        emb = get_embedding(txn_features, model)
        
        supabase.table("transactions").insert({
            "user_id": s['user_id'],
            "amount": round(np.random.uniform(2000, 5000), 2),
            "merchant": "High_Risk_Exchange",
            "location": "Unknown",
            "embedding": emb,
            "is_anomaly": True
        }).execute()

    print("✅ Seeding complete. Check your Streamlit Dashboard!")

if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_config = os.path.join(ROOT_DIR, "config.yaml")
    
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)