// ── Upload handling ────────────────────────────────────────────────────────
const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewWrap = document.getElementById("upload-preview-wrap");
const preview     = document.getElementById("upload-preview");
const runBtn      = document.getElementById("run-btn");
const resultsDiv  = document.getElementById("upload-results");

let selectedFile = null;

function showPreview(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  previewWrap.style.display = "block";
  runBtn.style.display = "block";
  resultsDiv.innerHTML = "";
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => { if (e.target.files[0]) showPreview(e.target.files[0]); });

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) showPreview(e.dataTransfer.files[0]);
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  runBtn.textContent = "Running…";
  runBtn.disabled = true;
  resultsDiv.innerHTML = "";

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const res  = await fetch("/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (data.error) { resultsDiv.innerHTML = `<p style="color:var(--red)">${data.error}</p>`; return; }

    if (data.detections.length === 0) {
      resultsDiv.innerHTML = `<p style="color:var(--muted);margin-top:10px">No objects detected above threshold.</p>`;
    } else {
      data.detections.forEach(d => {
        const el = document.createElement("div");
        el.className = "det-result";
        el.innerHTML = `
          ${d.crop_url ? `<img src="${d.crop_url}" alt="crop"/>` : ""}
          <div class="det-info">
            <strong>${d.class_name}</strong><br/>
            Confidence: ${(d.confidence * 100).toFixed(1)}%<br/>
            BBox: [${d.bbox.join(", ")}]
          </div>`;
        resultsDiv.appendChild(el);
      });
    }
    loadDetections();   // refresh table
  } catch(err) {
    resultsDiv.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
  } finally {
    runBtn.textContent = "Run Detection";
    runBtn.disabled = false;
  }
});

// ── Detection table ────────────────────────────────────────────────────────
async function loadDetections() {
  try {
    const res  = await fetch("/api/detections");
    const rows = await res.json();
    const body = document.getElementById("det-body");
    body.innerHTML = rows.map(d => `
      <tr>
        <td>${d.timestamp}</td>
        <td><span class="badge badge-${d.source === "webcam" ? "green" : "amber"}">${d.source}</span></td>
        <td>${d.class_name}</td>
        <td>${(d.confidence * 100).toFixed(1)}%</td>
        <td style="font-family:monospace;font-size:11px">[${d.bbox.join(",")}]</td>
        <td>${d.crop_url ? `<img class="crop-thumb" src="${d.crop_url}" alt="crop"/>` : "—"}</td>
      </tr>`).join("");
  } catch(e) { console.error(e); }
}

document.getElementById("refresh-btn")?.addEventListener("click", loadDetections);
loadDetections();
setInterval(loadDetections, 8000);   // auto-refresh every 8 s
