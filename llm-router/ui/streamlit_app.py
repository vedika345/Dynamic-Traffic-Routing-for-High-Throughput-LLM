"""
Simple Streamlit UI for the LLM Router.

Run with (in a separate terminal, while the API is running):
    streamlit run ui/streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="LLM Router Demo", page_icon="🧭")
st.title("🧭 LLM Router Demo")
st.write(
    "Type a prompt below. The router will classify it and pick the "
    "best local model (via Ollama) to handle it."
)

prompt = st.text_area("Your prompt", height=150, placeholder="Ask me anything...")

col1, col2 = st.columns(2)
send = col1.button("Send", type="primary")
show_stats = col2.button("Show routing stats")

if send and prompt.strip():
    with st.spinner("Routing and generating..."):
        try:
            resp = requests.post(f"{API_URL}/chat", json={"prompt": prompt}, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            st.markdown("### Response")
            st.write(data["response"])

            st.markdown("### Routing details")
            st.json({
                "task_type": data["task_type"],
                "complexity_score": round(data["complexity"], 3),
                "model_role": data["model_key"],
                "model_used": data["model_used"],
                "fallback_used": data["fallback_used"],
                "latency_sec": round(data["latency"], 2),
            })
        except Exception as e:
            st.error(f"Error talking to router API: {e}")

if show_stats:
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=10)
        resp.raise_for_status()
        st.markdown("### Routing statistics (all-time)")
        st.json(resp.json())
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
