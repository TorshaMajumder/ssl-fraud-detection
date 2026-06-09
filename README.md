# 🛡️ NeuralAudit: SSL-Powered Fraud Detection

**NeuralAudit** is a high-performance anomaly detection system designed to identify fraudulent financial transactions using **Self-Supervised Learning (SSL)** and **Vector Similarity Search**. Unlike rule-based systems, NeuralAudit learns a user's behavioral "representation" in a latent space, flagging zero-day fraud attempts that deviate from their unique mathematical "force field."

### 🚀 Key Technical Highlights
*   **Representational SSL:** Utilizes an SSL-trained bottleneck encoder to map 29-dimensional tabular transaction data as in [Kaggle Dataset: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into a 384-dimensional behavioral embedding space.
*   **Vector Engine:** Leverages **PostgreSQL with pgvector (via Supabase)** to store "Behavioral Anchors" and perform real-time Euclidean distance calculations.
*   **Human-in-the-loop (HITL) Feedback:** Features a dynamic thresholding system that allows human auditors to provide feedback, automatically tuning the model's sensitivity without code changes.
*   **GenAI Explainability:** Integrates **Llama-3.1 via Groq** to translate raw mathematical distances into plain-English security reports.

### 🛠️ The Stack
*   **Research/Training:** PyTorch, Scikit-Learn, Kaggle (GPU T4).
*   **Database:** Supabase (PostgreSQL + pgvector).
*   **Inference:** Groq (Llama-3.1-8b-instant).
*   **Frontend:** Streamlit.

### 🏗️ Technical Architecture
1.  **SSL Encoding:** An Autoencoder architecture trained on anonymized credit card data to capture non-linear feature relationships.
2.  **Behavioral Profiling:** Users are assigned an "Anchor" vector representing their centroid of normal behavior.
3.  **Anomaly Detection:** New transactions are encoded and compared against the Anchor using **Euclidean (L2) Distance**.
4.  **Threshold Adaptation:** Feedback from the UI triggers an adjustment in the `system_config` table, shifting the anomaly boundary based on False Positive/Negative reports.

### 📦 Setup & Installation
1. **Clone the repo:** `git clone https://github.com/TorshaMajumder/ssl-fraud-detection.git`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Database Setup:** 
   * Create a Supabase project and enable the `vector` extension.
   * Run the SQL scripts provided in `/database/schema.sql`.
4. **Environment Variables:** Set your `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `GROQ_API_KEY`.
5. **Run the Dashboard:** `streamlit run app.py`

---

## 🚀 Live Demo

<div align="center">
  <video src="YOUR_DRAGGED_LINK_HERE" width="100%" autoplay loop muted playsinline></video>

  <br/>

[Streamlit App](https://github.com/user-attachments/assets/29bc1796-3172-4b3d-944f-af0b1bc3962d)

</div>


---
