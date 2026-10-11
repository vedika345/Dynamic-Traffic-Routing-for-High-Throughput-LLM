"""
LLM Router — Professional Demo UI
Run with: streamlit run ui/dashboard.py
(Keep uvicorn running in another terminal first)
"""

import json
import os
import time
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000"
LOG_FILE = "logs/requests.jsonl"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Router",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── main header ── */
.main-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 60%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.main-header p { color: #8b949e; margin: 0; font-size: 0.95rem; }

/* ── stat cards ── */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label { font-size: 0.78rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── routing badge ── */
.route-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-code     { background: #1f4068; color: #58a6ff; border: 1px solid #388bfd; }
.badge-math     { background: #1f3a1f; color: #3fb950; border: 1px solid #2ea043; }
.badge-creative { background: #3d1f5e; color: #d2a8ff; border: 1px solid #a371f7; }
.badge-summary  { background: #3d2f00; color: #e3b341; border: 1px solid #d29922; }
.badge-chat     { background: #1f2d3d; color: #79c0ff; border: 1px solid #388bfd; }

/* ── response card ── */
.response-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 16px;
}
.response-card h4 { color: #8b949e; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; margin: 0 0 12px 0; }
.response-text { color: #e6edf3; line-height: 1.7; font-size: 0.95rem; white-space: pre-wrap; }
.code-block {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #e6edf3;
    overflow-x: auto;
    white-space: pre;
}

/* ── routing pipeline ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.pipeline-step {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #8b949e;
}
.pipeline-step span { color: #e6edf3; font-weight: 600; }
.pipeline-arrow { color: #388bfd; font-size: 1.1rem; }
.pipeline-step.highlight { border-color: #388bfd; background: #1a2332; }
.pipeline-step.highlight span { color: #58a6ff; }

/* ── complexity bar ── */
.complexity-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
}
.complexity-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* ── history item ── */
.history-item {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.history-prompt { color: #e6edf3; font-size: 0.9rem; margin-bottom: 8px; }
.history-meta { color: #8b949e; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }

/* ── model pill ── */
.model-pill {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #79c0ff;
}

/* ── inputs ── */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.15) !important;
}

/* ── buttons ── */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-size: 0.95rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2ea043 !important; }

/* ── divider ── */
hr { border-color: #30363d !important; }

/* ── selectbox / radio ── */
.stSelectbox > div > div, .stRadio > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* hide default streamlit decorations */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

TASK_BADGES = {
    "code":           ("badge-code",     "⚡ Code"),
    "math_reasoning": ("badge-math",     "∑ Math"),
    "creative":       ("badge-creative", "✦ Creative"),
    "summarization":  ("badge-summary",  "≡ Summary"),
    "general_chat":   ("badge-chat",     "💬 Chat"),
}

MODEL_LABELS = {
    "phi3:mini":         ("Simple Chat",    "#79c0ff"),
    "llama3.2:3b":       ("General",        "#56d364"),
    "llama3.1:8b":       ("Deep Reasoning", "#3fb950"),
    "qwen2.5-coder:7b":  ("Code Expert",    "#58a6ff"),
    "mistral:7b":        ("Creative",       "#d2a8ff"),
}

def complexity_color(score):
    if score < 0.35:   return "#3fb950"
    elif score < 0.65: return "#e3b341"
    else:              return "#f85149"

def complexity_label(score):
    if score < 0.35:   return "Simple"
    elif score < 0.65: return "Moderate"
    else:              return "Complex"

def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try: entries.append(json.loads(line))
                except: pass
    return entries

def check_server():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 LLM Router")
    st.markdown("---")

    server_ok = check_server()
    if server_ok:
        st.markdown("🟢 **API server online**")
    else:
        st.markdown("🔴 **API server offline**")
        st.caption("Run `uvicorn app.main:app --reload` in another terminal.")

    st.markdown("---")
    st.markdown("#### Routing Table")
    st.markdown("""
| Task | Model |
|------|-------|
| 💬 Simple chat | phi3:mini |
| 📖 General | llama3.2:3b |
| ∑ Hard math | llama3.1:8b |
| ⚡ Code | qwen2.5-coder |
| ✦ Creative | mistral:7b |
""")

    st.markdown("---")
    st.markdown("#### Quick Prompts")
    quick = st.selectbox("Load example:", [
        "— select —",
        "Write a Python function to sort a list using merge sort.",
        "What is the capital of Japan?",
        "Derive the quadratic formula step by step.",
        "Write a short poem about a rainy evening.",
        "Summarize what machine learning is in 3 sentences.",
        "Hello! How are you doing today?",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='color:#8b949e;font-size:0.78rem'>LLM Router · Local AI</div>", unsafe_allow_html=True)

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧭 LLM Router</h1>
  <p>Classifies every prompt by task type &amp; complexity, then routes it to the best local model — automatically.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────────────────────────────
log_entries = load_log()
total = len(log_entries)
fallbacks = sum(1 for e in log_entries if e.get("fallback_used"))
avg_lat = (sum(e.get("total_latency", 0) for e in log_entries) / total) if total else 0
models_used = len(set(e.get("model_used","") for e in log_entries))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{total}</div><div class="stat-label">Total Requests</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{models_used}</div><div class="stat-label">Models Active</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{fallbacks}</div><div class="stat-label">Fallbacks Used</div></div>', unsafe_allow_html=True)
with c4:
    lat_display = f"{avg_lat:.1f}s" if total else "—"
    st.markdown(f'<div class="stat-card"><div class="stat-number">{lat_display}</div><div class="stat-label">Avg Latency</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two column layout ──────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown("#### Send a Prompt")

    default_text = quick if quick != "— select —" else ""
    prompt = st.text_area(
        "prompt",
        value=default_text,
        height=130,
        placeholder="Ask anything — code, math, creative writing, general questions...",
        label_visibility="collapsed",
    )

    send_btn = st.button("Route & Generate →", use_container_width=True)

    if send_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt first.")
        elif not server_ok:
            st.error("API server is offline. Start it with: `uvicorn app.main:app --reload`")
        else:
            with st.spinner("Classifying → routing → generating..."):
                try:
                    start = time.time()
                    resp = requests.post(f"{API_URL}/chat", json={"prompt": prompt}, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    elapsed = time.time() - start
                    st.session_state.history.insert(0, {"prompt": prompt, "data": data})
                except requests.exceptions.Timeout:
                    st.error("Request timed out (model is slow on CPU). Try a simpler prompt first.")
                    data = None
                except Exception as e:
                    st.error(f"Error: {e}")
                    data = None

            if data:
                task = data["task_type"]
                badge_cls, badge_label = TASK_BADGES.get(task, ("badge-chat","💬 Chat"))
                comp = data["complexity"]
                model = data["model_used"]
                model_role, model_color = MODEL_LABELS.get(model, ("Model", "#8b949e"))

                # ── Routing pipeline ──
                st.markdown("**Routing Decision**")
                st.markdown(f"""
<div class="pipeline">
  <div class="pipeline-step">Prompt<br><span>{prompt[:30]}{'...' if len(prompt)>30 else ''}</span></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">Task Type<br><span class="route-badge {badge_cls}">{badge_label}</span></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step">Complexity<br><span style="color:{complexity_color(comp)}">{complexity_label(comp)} ({comp:.2f})</span></div>
  <div class="pipeline-arrow">→</div>
  <div class="pipeline-step highlight">Model<br><span class="model-pill">{model}</span></div>
</div>
""", unsafe_allow_html=True)

                # ── Complexity bar ──
                bar_pct = int(comp * 100)
                bar_color = complexity_color(comp)
                st.markdown(f"""
<div style="margin: 4px 0 16px 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#8b949e;margin-bottom:4px;">
    <span>Complexity Score</span><span style="color:{bar_color}">{comp:.3f}</span>
  </div>
  <div class="complexity-bar-bg">
    <div class="complexity-bar-fill" style="width:{bar_pct}%;background:{bar_color}"></div>
  </div>
</div>
""", unsafe_allow_html=True)

                # ── Metrics row ──
                m1, m2, m3 = st.columns(3)
                m1.metric("Model Role", data["model_key"].replace("_"," ").title())
                m2.metric("Latency", f"{data['latency']:.1f}s")
                m3.metric("Fallback", "Yes ⚠️" if data["fallback_used"] else "No ✓")

                # ── Response ──
                st.markdown("**Model Response**")
                response_text = data["response"]
                if "```" in response_text:
                    parts = response_text.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            if part.strip():
                                st.markdown(f'<div class="response-card"><p class="response-text">{part.strip()}</p></div>', unsafe_allow_html=True)
                        else:
                            lines = part.split("\n")
                            code = "\n".join(lines[1:]) if lines[0].strip() in ["python","js","bash","json",""] else part
                            st.markdown(f'<div class="code-block">{code}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="response-card"><p class="response-text">{response_text}</p></div>', unsafe_allow_html=True)

with right:
    st.markdown("#### Routing History")

    if not st.session_state.history:
        st.markdown('<div class="response-card"><p style="color:#8b949e;text-align:center;margin:24px 0">No requests yet.<br>Send a prompt to see routing history.</p></div>', unsafe_allow_html=True)
    else:
        for item in st.session_state.history[:8]:
            d = item["data"]
            task = d["task_type"]
            badge_cls, badge_label = TASK_BADGES.get(task, ("badge-chat","💬"))
            comp = d["complexity"]
            fb = " · ⚠️ fallback" if d["fallback_used"] else ""
            st.markdown(f"""
<div class="history-item">
  <div class="history-prompt">{item['prompt'][:80]}{'...' if len(item['prompt'])>80 else ''}</div>
  <div class="history-meta">
    <span class="route-badge {badge_cls}">{badge_label}</span>&nbsp;&nbsp;
    <span class="model-pill">{d['model_used']}</span>&nbsp;&nbsp;
    <span style="color:{complexity_color(comp)}">{complexity_label(comp)}</span>&nbsp;·&nbsp;
    {d['latency']:.1f}s{fb}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Model Usage")

    if log_entries:
        from collections import Counter
        model_counts = Counter(e["model_used"] for e in log_entries)
        task_counts  = Counter(e["task_type"]  for e in log_entries)

        st.markdown("**By model**")
        for model, count in model_counts.most_common():
            pct = int(count / total * 100)
            _, color = MODEL_LABELS.get(model, ("", "#8b949e"))
            st.markdown(f"""
<div style="margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px">
    <span style="color:#e6edf3;font-family:'JetBrains Mono',monospace">{model}</span>
    <span style="color:#8b949e">{count} ({pct}%)</span>
  </div>
  <div class="complexity-bar-bg">
    <div class="complexity-bar-fill" style="width:{pct}%;background:{color}"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("**By task type**")
        for task, count in task_counts.most_common():
            pct = int(count / total * 100)
            badge_cls, badge_label = TASK_BADGES.get(task, ("badge-chat", task))
            st.markdown(f"""
<div style="margin-bottom:6px;display:flex;justify-content:space-between;align-items:center">
  <span class="route-badge {badge_cls}">{badge_label}</span>
  <span style="color:#8b949e;font-size:0.82rem">{count} requests</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#8b949e;font-size:0.85rem">Stats appear after first request.</p>', unsafe_allow_html=True)
