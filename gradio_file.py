import os
import requests
import gradio as gr
import yaml

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

API_URL = os.getenv("API_URL", f"http://localhost:{cfg['server']['api_port']}")

def discover(query, k):
    if not query or not query.strip():
        return "Please enter a description.", []

    r = requests.post(f"{API_URL}/retrieve", json={"query": query, "k": int(k)}, timeout=20)
    if r.status_code != 200:
        return f"API error: {r.text}", []

    results = r.json().get("results", [])
    gallery = []
    for x in results:
        img = x.get("poster_url")  # URL served by Flask
        cap = f"{x.get('title','')} | {x.get('genre','')} | match={x.get('matched_on')}"
        if img:
            gallery.append((img, cap))

    md = "### 🎯 Results\n" + "\n".join([f"- **{x.get('title','')}** ({x.get('genre','')})" for x in results])
    return md, gallery

def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return f"✅ API OK — {r.json()}"
    except Exception:
        pass
    return "❌ API not reachable"

with gr.Blocks(title="Part 4 — Natural Language Movie Discovery") as demo:
    gr.Markdown("# 🎬 Part 4 — Natural Language Movie Discovery (CLIP + Annoy)")
    status = gr.Textbox(label="API status", value=api_health(), interactive=False)
    gr.Button("Refresh").click(fn=api_health, outputs=status)

    with gr.Tabs():
        with gr.TabItem("💬 Discovery (Text → Posters)"):
            q = gr.Textbox(lines=2, label="Describe what you want", placeholder="e.g., a masked killer in a small town")
            k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            btn = gr.Button("Search", variant="primary")
            out_md = gr.Markdown()
            gallery = gr.Gallery(label="Matches", columns=5, height=300)
            btn.click(fn=discover, inputs=[q, k], outputs=[out_md, gallery])

if __name__ == "__main__":
    host = cfg["server"]["gradio_host"]
    port = int(cfg["server"]["gradio_port"])
    print(f"Gradio running on http://{host}:7861 (API={API_URL})")
    demo.launch(server_name=host, server_port=7861, share=False)
