import os
import yaml
import pandas as pd
from groq import Groq
import streamlit as st
from supabase import create_client


def apply_human_feedback(txn_id, was_actually_fraud, supabase):
    # Fetch current threshold
    res = supabase.table("system_config").select("value").eq("key", "anomaly_threshold").execute()
    current_threshold = res.data[0]['value']
    
    # HF Logic: If the user says it was a "False Alarm", we increase the threshold
    if not was_actually_fraud:
        new_threshold = current_threshold * 1.1 # Relax the threshold by 10%
        supabase.table("system_config").update({"value": new_threshold}).eq("key", "anomaly_threshold").execute()
        st.success(f"Threshold updated to {new_threshold:.2f} (Relaxed due to False Positive)")
    else:
        st.info("System confirmed. No threshold change needed.")
    
    # Mark the transaction as reviewed
    supabase.table("transactions").update({"human_feedback": was_actually_fraud}).eq("id", txn_id).execute()


def main(config):

    st.set_page_config(page_title="The Financial Guardian", layout="wide")
    st.title("🛡️ The Financial Guardian")
    st.subheader("Real-time SSL Anomaly Detection & Human Feedback Loop")


    supabase = create_client(config["supabase_url"], config["supabase_key"])
    groq_client = Groq(api_key=config["GROQ_API_KEY"])

    # --- UI LAYOUT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### 🚨 Recent Transactions")
        # Fetch recent transactions
        txns = supabase.table("transactions").select("*").order("created_at", desc=True).limit(10).execute()
        
        if txns.data:
            df = pd.DataFrame(txns.data)
            # Display table with red highlighting for anomalies
            def highlight_anomaly(s):
                return ['background-color: #ff4b4b' if s.is_anomaly else '' for _ in s]
            
            st.dataframe(df[['id', 'amount', 'merchant', 'location', 'is_anomaly']].style.apply(highlight_anomaly, axis=1))

            # --- HUMAN FEEDBACK SECTION ---
            st.write("### ✍️ Review Anomalies")
            anomaly_list = [t for t in txns.data if t['is_anomaly'] and t['human_feedback'] is None]
            
            if anomaly_list:
                selected_id = st.selectbox("Select Anomaly ID to Review", [t['id'] for t in anomaly_list])
                selected_txn = next(t for t in anomaly_list if t['id'] == selected_id)
                
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("✅ Confirm Fraud", use_container_width=True):
                    apply_human_feedback(selected_id, True, supabase)
                    st.rerun()
                if c_btn2.button("❌ False Alarm", use_container_width=True):
                    apply_human_feedback(selected_id, False, supabase)
                    st.rerun()
            else:
                st.write("No new anomalies to review.")

    with col2:
        st.write("### 🤖 Analyst Reasoning (GenAI)")
        if 'selected_id' in locals() and anomaly_list:
            if st.button("Explain this Anomaly"):
                with st.spinner("Analyzing Latent Space..."):
                    # Use Groq to explain the concept
                    prompt = f"""
                    Analyze this transaction: {selected_txn['merchant']} for ${selected_txn['amount']}.
                    The Self Supervised learning (SSL) model flagged it because its Euclidean distance from the user's 
                    behavioral anchor was significant. 
                    Explain why behavioral-based detection is better than simple rules for this case.
                    """
                    response = groq_client.chat.completions.create(
                        model=config['groq_model_name'],
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.info(response.choices[0].message.content)
        
        st.divider()
        # Show current threshold
        res = supabase.table("system_config").select("value").eq("key", "anomaly_threshold").execute()
        st.metric("Current Anomaly Threshold", f"{res.data[0]['value']:.2f}")

if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_to_config = os.path.join(ROOT_DIR, "config.yaml")
    
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)