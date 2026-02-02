import gradio as gr
import requests
import io
import os
from PIL import Image
import yaml


CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

API_URL = os.getenv("API_URL", f"http://127.0.0.1:{cfg['server']['api_port']}")


def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return "✅ API disponible"
        return f"❌ API erreur: {r.status_code}"
    except Exception:
        return "❌ API non disponible"


def text_search(query, k):
    if not query or not query.strip():
        return "Please enter a description.", []

    try:
        r = requests.post(f"{API_URL}/retrieve", json={"query": query, "k": int(k)}, timeout=20)
    except requests.exceptions.ConnectionError as e:
        return f"❌ Cannot reach API at {API_URL}\n\n{e}", []
    except Exception as e:
        return f"❌ Unexpected error:\n\n{e}", []

    if r.status_code != 200:
        return f"❌ API error {r.status_code}:\n\n{r.text}", []

    results = r.json().get("results", [])
    md = "### 🎯 Results\n" + "\n".join(
        [f"- **{x.get('title','')}** ({x.get('genre','')}) — match: {x.get('matched_on')}"
         for x in results]
    )

    gallery = []
    for x in results:
        url = x.get("poster_url")
        cap = f"{x.get('title','')} | {x.get('genre','')} | match={x.get('matched_on')}"
        if url:
            gallery.append((url, cap))

    return md, gallery


def image_search(image, k):
    if image is None:
        return "Please upload an image.", []

    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        files = {"file": ("query.jpg", img_bytes.getvalue(), "image/jpeg")}

        r = requests.post(f"{API_URL}/retrieve_image", files=files, data={"k": int(k)}, timeout=30)
    except requests.exceptions.ConnectionError as e:
        return f"❌ Cannot reach API at {API_URL}\n\n{e}", []
    except Exception as e:
        return f"❌ Unexpected error:\n\n{e}", []

    if r.status_code != 200:
        return f"❌ API error {r.status_code}:\n\n{r.text}", []

    results = r.json().get("results", [])
    md = "### 🖼️ Image Search Results\n" + "\n".join(
        [f"- **{x.get('title','')}** ({x.get('genre','')}) — match: {x.get('matched_on')}"
         for x in results]
    )

    gallery = []
    for x in results:
        url = x.get("poster_url")
        cap = f"{x.get('title','')} | {x.get('genre','')} | match={x.get('matched_on')}"
        if url:
            gallery.append((url, cap))

    return md, gallery

def chat_send(history, message, k):
    history = history or []
    if not message or not message.strip():
        return history, "", []

    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"message": message, "k": int(k)},
            timeout=30
        )
        if r.status_code != 200:
            answer = f"❌ API error {r.status_code}: {r.text}"
            gallery = []
        else:
            data = r.json()
            answer = data.get("answer", "")
            # Construire la galerie avec les posters
            results = data.get("results", [])
            gallery = []
            for res in results:
                url = res.get("poster_url")
                title = res.get("title", "Unknown")
                genre = res.get("genre", "N/A")
                if url:
                    gallery.append((url, f"{title} ({genre})"))
    except Exception as e:
        answer = f"❌ API not reachable: {e}"
        gallery = []

    # FORMAT GRADIO 6.0: list of messages with role/content
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, "", gallery



with gr.Blocks(title="Part 4 — Movie Discovery (CLIP + Annoy)") as demo:
    gr.Markdown("# 🎬 Part 4 — Natural Language Movie Discovery (CLIP + single Annoy)")

    with gr.Row():
        api_status = gr.Textbox(label="🔌 API Status", value=check_api_health(), interactive=False)
        refresh_btn = gr.Button("🔄 Refresh")
    refresh_btn.click(fn=check_api_health, outputs=api_status)

    with gr.Tabs():
        with gr.TabItem("💬 Discovery (Text → Posters)"):
            query = gr.Textbox(
                lines=2,
                label="Describe what you want",
                placeholder="e.g. a masked killer, a love story in Paris, alien spaceship..."
            )
            k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            btn = gr.Button("Search", variant="primary")

            out_md = gr.Markdown()
            gallery = gr.Gallery(label="Matches", columns=5, height=320)

            btn.click(fn=text_search, inputs=[query, k], outputs=[out_md, gallery])

        with gr.TabItem("🖼️ Discovery (Image → Posters)"):
            img = gr.Image(type="pil", label="Upload an image", height=320)
            k2 = gr.Slider(1, 10, value=5, step=1, label="Top K")
            btn2 = gr.Button("Search by image", variant="primary")

            out_md2 = gr.Markdown()
            gallery2 = gr.Gallery(label="Matches", columns=5, height=320)

            btn2.click(fn=image_search, inputs=[img, k2], outputs=[out_md2, gallery2])

        with gr.TabItem("🤖 Chat (RAG)"):
            chatbot = gr.Chatbot()  # <-- PAS de type ici
            msg = gr.Textbox(label="Message", placeholder="I want a thriller with a masked killer")
            k3 = gr.Slider(1, 10, value=5, step=1, label="Top K")
            send = gr.Button("Send", variant="primary")
            
            # Galerie pour afficher les posters des résultats
            chat_gallery = gr.Gallery(label="Recommended Movies", columns=5, height=320)

            send.click(fn=chat_send, inputs=[chatbot, msg, k3], outputs=[chatbot, msg, chat_gallery])




if __name__ == "__main__":
    print(f"Gradio running on http://0.0.0.0:{cfg['server']['gradio_port']} (API={API_URL})")
    demo.launch(
        server_name=cfg["server"]["gradio_host"],
        server_port=int(cfg["server"]["gradio_port"]),
        share=False,
        theme=gr.themes.Soft()
    )
