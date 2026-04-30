"""
uploader.py — Local web UI for adding PDFs or URLs to the CSE-GO knowledge pool.

Usage:
    python3 uploader.py
    Then open http://localhost:7860 in your browser.
"""

import hashlib
import html as html_module
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from html.parser import HTMLParser
from queue import Queue

import fitz
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

CHROMA_DIR  = "./chroma-db"
COLLECTION  = "cse_knowledge_base"
CHUNK_WORDS = 800
OVERLAP     = 150
EMBED_MODEL = "all-MiniLM-L6-v2"

SUBJECTS = [
    "Auto-detect",
    "History", "Geography", "Polity", "Economics",
    "Science & Technology", "Current Affairs",
    "Art & Culture", "Environment", "General",
]

# ── HTML PAGE ─────────────────────────────────────────────────────────────────

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSE-GO Knowledge Uploader</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f0f0f; color: #e8e8e8; min-height: 100vh;
         display: flex; align-items: center; justify-content: center; padding: 24px; }
  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px;
          padding: 32px; width: 100%; max-width: 560px; }
  h1 { font-size: 1.25rem; font-weight: 600; margin-bottom: 6px; color: #fff; }
  .subtitle { font-size: 0.82rem; color: #666; margin-bottom: 28px; }

  .tabs { display: flex; gap: 8px; margin-bottom: 24px; }
  .tab { flex: 1; padding: 8px; border: 1px solid #2a2a2a; border-radius: 8px;
         background: transparent; color: #888; cursor: pointer; font-size: 0.85rem;
         transition: all 0.15s; }
  .tab.active { background: #2563eb; border-color: #2563eb; color: #fff; }

  .panel { display: none; }
  .panel.active { display: block; }

  label { font-size: 0.78rem; color: #aaa; display: block; margin-bottom: 6px; margin-top: 16px; }
  label:first-child { margin-top: 0; }

  input[type=text], input[type=number], select {
    width: 100%; padding: 9px 12px; background: #111; border: 1px solid #2a2a2a;
    border-radius: 8px; color: #e8e8e8; font-size: 0.875rem; outline: none;
    transition: border-color 0.15s; }
  input:focus, select:focus { border-color: #2563eb; }

  .drop-zone { border: 2px dashed #2a2a2a; border-radius: 10px; padding: 32px;
               text-align: center; cursor: pointer; transition: all 0.15s;
               background: #111; margin-bottom: 0; }
  .drop-zone:hover, .drop-zone.drag { border-color: #2563eb; background: #12203e; }
  .drop-zone p { color: #555; font-size: 0.85rem; margin-top: 6px; }
  .drop-zone .icon { font-size: 1.8rem; }
  .drop-zone .filename { color: #2563eb; font-size: 0.85rem; margin-top: 6px;
                          font-weight: 500; word-break: break-all; }
  #pdf-input { display: none; }

  .row { display: flex; gap: 12px; }
  .row > div { flex: 1; }

  button[type=submit] {
    width: 100%; margin-top: 24px; padding: 11px;
    background: #2563eb; border: none; border-radius: 8px;
    color: #fff; font-size: 0.9rem; font-weight: 600;
    cursor: pointer; transition: background 0.15s; }
  button[type=submit]:hover { background: #1d4ed8; }
  button[type=submit]:disabled { background: #1e3a6e; color: #555; cursor: not-allowed; }

  #log-box { display: none; margin-top: 20px; background: #0a0a0a;
             border: 1px solid #2a2a2a; border-radius: 8px;
             padding: 14px 16px; font-family: monospace; font-size: 0.78rem;
             color: #a0c070; max-height: 280px; overflow-y: auto;
             white-space: pre-wrap; word-break: break-word; line-height: 1.6; }
  .done { color: #4ade80; font-weight: bold; }
  .err  { color: #f87171; font-weight: bold; }
  .warn { color: #fbbf24; }
</style>
</head>
<body>
<div class="card">
  <h1>CSE-GO Knowledge Uploader</h1>
  <p class="subtitle">Add PDFs or web articles to the RAG knowledge pool</p>

  <div class="tabs">
    <button class="tab active" onclick="switchTab('pdf')">📄 PDF</button>
    <button class="tab" onclick="switchTab('url')">🔗 URL</button>
  </div>

  <form id="upload-form" onsubmit="submitForm(event)">

    <!-- PDF panel -->
    <div class="panel active" id="panel-pdf">
      <label>PDF File</label>
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('pdf-input').click()"
           ondragover="event.preventDefault();this.classList.add('drag')"
           ondragleave="this.classList.remove('drag')"
           ondrop="handleDrop(event)">
        <div class="icon">📂</div>
        <p>Click to browse or drag &amp; drop a PDF</p>
        <div class="filename" id="filename-display"></div>
      </div>
      <input type="file" id="pdf-input" accept=".pdf" onchange="handleFile(this.files[0])">

      <div class="row">
        <div>
          <label>Start page <span style="color:#555">(optional)</span></label>
          <input type="number" id="start-page" name="start_page" min="1" placeholder="1">
        </div>
        <div>
          <label>End page <span style="color:#555">(optional)</span></label>
          <input type="number" id="end-page" name="end_page" min="1" placeholder="last">
        </div>
      </div>
    </div>

    <!-- URL panel -->
    <div class="panel" id="panel-url">
      <label>URL</label>
      <input type="text" id="url-input" name="url" placeholder="https://example.com/article">
    </div>

    <!-- Shared fields -->
    <label style="margin-top:20px">Subject</label>
    <select name="subject" id="subject-select">
      {% for s in subjects %}
      <option value="{{ s }}">{{ s }}</option>
      {% endfor %}
    </select>

    <button type="submit" id="submit-btn">Add to Knowledge Pool</button>
  </form>

  <div id="log-box"></div>
</div>

<script>
let activeTab = 'pdf';
let selectedFile = null;

function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', ['pdf','url'][i] === tab));
  document.getElementById('panel-pdf').classList.toggle('active', tab === 'pdf');
  document.getElementById('panel-url').classList.toggle('active', tab === 'url');
}

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  document.getElementById('filename-display').textContent = file.name;
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.pdf')) handleFile(file);
}

async function submitForm(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  const log = document.getElementById('log-box');
  log.innerHTML = '';
  log.style.display = 'block';
  btn.disabled = true;
  btn.textContent = 'Processing…';

  const subject = document.getElementById('subject-select').value;

  const fd = new FormData();
  fd.append('subject', subject);

  if (activeTab === 'pdf') {
    if (!selectedFile) { appendLog('⚠ Please select a PDF file.', 'warn'); btn.disabled=false; btn.textContent='Add to Knowledge Pool'; return; }
    fd.append('pdf', selectedFile);
    const sp = document.getElementById('start-page').value;
    const ep = document.getElementById('end-page').value;
    if (sp) fd.append('start_page', sp);
    if (ep) fd.append('end_page', ep);
  } else {
    const url = document.getElementById('url-input').value.trim();
    if (!url) { appendLog('⚠ Please enter a URL.', 'warn'); btn.disabled=false; btn.textContent='Add to Knowledge Pool'; return; }
    fd.append('url', url);
  }

  try {
    const resp = await fetch('/ingest', { method: 'POST', body: fd });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      dec.decode(value).split('\\n').forEach(line => {
        if (line.startsWith('data:')) {
          const msg = line.slice(5).trim();
          const cls = msg.startsWith('✓') ? 'done' : msg.startsWith('ERROR') || msg.startsWith('✗') ? 'err' : msg.startsWith('WARNING') || msg.startsWith('Compressing') || msg.startsWith('Uploading') || msg.startsWith('Syncing') ? 'warn' : '';
          appendLog(msg, cls);
        }
      });
    }
  } catch(err) {
    appendLog('ERROR: ' + err.message, 'err');
  }

  btn.disabled = false;
  btn.textContent = 'Add to Knowledge Pool';
  // Reset subject dropdown and file selection for next upload
  document.getElementById('subject-select').value = 'Auto-detect';
  selectedFile = null;
  document.getElementById('filename-display').textContent = '';
  document.getElementById('pdf-input').value = '';
  document.getElementById('url-input').value = '';
  document.getElementById('start-page').value = '';
  document.getElementById('end-page').value = '';
}

function appendLog(msg, cls) {
  const log = document.getElementById('log-box');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = msg;
  log.appendChild(span);
  log.appendChild(document.createTextNode('\\n'));
  log.scrollTop = log.scrollHeight;
}
</script>
</body>
</html>"""


# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────

def extract_pdf_pages(path, start=1, end=None):
    doc = fitz.open(path)
    total = len(doc)
    end = min(total, end or total)
    pages = []
    for i in range(len(doc)):
        pn = i + 1
        if pn < start or pn > end:
            continue
        text = doc[i].get_text().strip()
        if text:
            pages.append((pn, text))
    doc.close()
    return pages, total


class _TextExtractor(HTMLParser):
    SKIP_TAGS = {"script","style","nav","footer","header","aside",
                 "noscript","form","button","svg","figure"}
    def __init__(self):
        super().__init__()
        self.chunks = []; self._skip = 0
    def handle_starttag(self, tag, _):
        if tag.lower() in self.SKIP_TAGS: self._skip += 1
    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS: self._skip = max(0, self._skip - 1)
    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t: self.chunks.append(t)
    def get_text(self): return " ".join(self.chunks)


def extract_url_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html_module.unescape(raw))
    text = re.sub(r'\s+', ' ', parser.get_text()).strip()
    return text


def infer_subject(name):
    n = name.lower()
    if any(x in n for x in ("history","past","heritage","medieval","ancient")): return "History"
    if any(x in n for x in ("geography","climate","river","soil","geograph")):  return "Geography"
    if any(x in n for x in ("polity","constitution","governance","civics","political","parliament")): return "Polity"
    if any(x in n for x in ("economy","economics","finance","budget","gdp","trade")): return "Economics"
    if any(x in n for x in ("science","technology","biology","chemistry","physics","space")): return "Science & Technology"
    if any(x in n for x in ("current","news","affairs","monthly","weekly")): return "Current Affairs"
    if any(x in n for x in ("environment","ecology","wildlife","forest")): return "Environment"
    if any(x in n for x in ("art","culture","dance","music","festival","painting")): return "Art & Culture"
    return "General"


def chunk_pages(pages):
    word_list = []
    for pn, text in pages:
        for w in text.split():
            word_list.append((w, pn))
    chunks, step, i = [], CHUNK_WORDS - OVERLAP, 0
    while i < len(word_list):
        window = word_list[i: i + CHUNK_WORDS]
        chunks.append({"text": " ".join(w for w, _ in window),
                       "start_page": window[0][1], "chunk_index": len(chunks)})
        i += step
    return chunks


# ── FLASK APP ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB


@app.route("/")
def index():
    return render_template_string(PAGE, subjects=SUBJECTS)


@app.route("/ingest", methods=["POST"])
def ingest():
    def stream():
        subject_arg = request.form.get("subject", "Auto-detect")
        tmp_path = None

        try:
            # ── PDF branch ────────────────────────────────────────────────────
            if "pdf" in request.files:
                f = request.files["pdf"]
                source_name = f.filename
                start = int(request.form.get("start_page") or 1)
                end   = int(request.form.get("end_page") or 0) or None

                yield f"data: Source: {source_name} (PDF)\n\n"

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                f.save(tmp.name)
                tmp_path = tmp.name

                pages, total_pages = extract_pdf_pages(tmp_path, start=start, end=end)
                end_display = end or total_pages
                page_info = f"pages {start}–{end_display} of {total_pages}" if (start > 1 or (end and end < total_pages)) else f"all {total_pages} pages"
                yield f"data: Pages: {page_info}  ({len(pages)} with text)\n\n"

            # ── URL branch ────────────────────────────────────────────────────
            elif "url" in request.form:
                source_name = request.form["url"].strip()
                yield f"data: Source: {source_name} (URL)\n\n"
                yield "data: Fetching page...\n\n"
                text = extract_url_text(source_name)
                pages = [(1, text)]
                page_info = "1 page (web)"
                yield f"data: Fetched {len(text.split()):,} words\n\n"

            else:
                yield "data: ERROR: No PDF or URL provided.\n\n"
                return

            if not pages:
                yield "data: ERROR: No text could be extracted.\n\n"
                return

            total_words = sum(len(p[1].split()) for p in pages)

            # ── Subject ───────────────────────────────────────────────────────
            if subject_arg and subject_arg != "Auto-detect":
                subject = subject_arg
            else:
                subject = infer_subject(source_name)
                yield f"data: Subject: {subject} (auto-detected)\n\n"

            # ── Chunk ─────────────────────────────────────────────────────────
            chunks = chunk_pages(pages)
            yield f"data: Words: {total_words:,}  |  Chunks: {len(chunks)}\n\n"

            # ── ChromaDB ──────────────────────────────────────────────────────
            yield "data: Connecting to ChromaDB...\n\n"
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_or_create_collection(name=COLLECTION, embedding_function=ef)
            before = collection.count()

            # ── Upsert in batches ─────────────────────────────────────────────
            source_hash = hashlib.md5(source_name.encode()).hexdigest()[:8]
            documents, metadatas, ids = [], [], []
            for c in chunks:
                documents.append(c["text"])
                metadatas.append({"source": source_name, "page": c["start_page"],
                                  "chunk_index": c["chunk_index"], "subject": subject})
                ids.append(f"{source_hash}_p{c['start_page']}_c{c['chunk_index']}")

            yield "data: Embedding and storing (slow step — updates every 25 chunks)...\n\n"
            batch = 25
            for i in range(0, len(chunks), batch):
                collection.upsert(
                    documents=documents[i:i+batch],
                    metadatas=metadatas[i:i+batch],
                    ids=ids[i:i+batch],
                )
                done = min(i + batch, len(chunks))
                yield f"data: Stored {done}/{len(chunks)} chunks...\n\n"

            after  = collection.count()
            net    = after - before
            mb     = sum(len(d.encode()) for d in documents) / (1024 * 1024)

            yield f"data: ─────────────────────────────\n\n"
            yield f"data: Ingest complete. Syncing to cloud...\n\n"

            # ── Push updated chroma-db to Cloudflare R2 ───────────────────────
            try:
                yield "data: Compressing chroma-db...\n\n"
                tar_result = subprocess.run(
                    ["tar", "-czf", "chroma-db.tar.gz", "chroma-db/"],
                    capture_output=True, text=True
                )
                if tar_result.returncode != 0:
                    raise RuntimeError(tar_result.stderr.strip())

                yield "data: Uploading to Cloudflare R2...\n\n"
                rclone_result = subprocess.run(
                    ["rclone", "copy", "chroma-db.tar.gz", "r2:cse-go-pipeline/"],
                    capture_output=True, text=True
                )
                if rclone_result.returncode != 0:
                    raise RuntimeError(rclone_result.stderr.strip())

                yield "data: ✓ Synced to R2 — source is now quiz-ready\n\n"
            except FileNotFoundError:
                yield "data: WARNING: rclone not found — skipping R2 sync. Run manually: rclone copy chroma-db.tar.gz r2:cse-go-pipeline/\n\n"
            except Exception as sync_err:
                yield f"data: WARNING: R2 sync failed: {sync_err}\n\n"

            yield f"data: ─────────────────────────────\n\n"
            yield f"data: ✓ QUIZ-READY\n\n"
            yield f"data: Subject         : {subject}\n\n"
            yield f"data: Pages ingested  : {page_info}\n\n"
            yield f"data: Words ingested  : {total_words:,}\n\n"
            yield f"data: Text size       : {mb:.2f} MB\n\n"
            yield f"data: Chunks stored   : {len(chunks)}  (net new: {net:+d})\n\n"
            yield f"data: Collection total: {after:,} chunks\n\n"

        except Exception as e:
            yield f"data: ERROR: {e}\n\n"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache",
                             "Transfer-Encoding": "chunked"})


if __name__ == "__main__":
    print("\nCSE-GO Knowledge Uploader")
    print("Open http://localhost:7860 in your browser\n")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
