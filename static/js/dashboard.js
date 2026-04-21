let detections = [];
let selected = null;

const listEl = document.getElementById("detectionList");

const emptyState = document.getElementById("emptyState");
const detailPanel = document.getElementById("detailPanel");

// fields
const d_class = document.getElementById("d_class");
const d_conf = document.getElementById("d_conf");
const d_cam = document.getElementById("d_cam");
const d_time = document.getElementById("d_time");

const cropImg = document.getElementById("cropImg");
const heatmapImg = document.getElementById("heatmapImg");
const frameImg = document.getElementById("frameImg");

const heatmapFallback = document.getElementById("heatmapFallback");

// buttons
const acceptBtn = document.getElementById("acceptBtn");
const rejectBtn = document.getElementById("rejectBtn");

// FETCH LOOP
async function fetchData() {
  try {
    const res = await fetch("/api/detections");
    detections = await res.json();
    renderList();
    updateStats();
  } catch (err) {
    console.error(err);
  }
}

// STATS
function updateStats() {
  document.getElementById("totalDetections").innerText = detections.length;
  document.getElementById("alertsToday").innerText = detections.length;
}

// LIST
function renderList() {
  listEl.innerHTML = "";

  detections.forEach(d => {
    const li = document.createElement("li");
    li.className = "detection-item";

    if (selected && selected.id === d.id) {
      li.classList.add("active");
    }

    li.onclick = () => selectItem(d);

    li.innerHTML = `
      ${d.crop_url ? `<img class="thumb" src="${d.crop_url}"/>` : ""}
      <div>
        <div>${d.class_name} (${d.confidence})</div>
        <div class="meta">Cam ${d.camera_id} • ${d.timestamp}</div>
      </div>
    `;

    listEl.appendChild(li);
  });
}

// SELECT
function selectItem(d) {
  selected = d;

  emptyState.style.display = "none";
  detailPanel.classList.remove("hidden");

  d_class.innerText = d.class_name;
  d_conf.innerText = d.confidence;
  d_cam.innerText = d.camera_id;
  d_time.innerText = d.timestamp;

  // images
  cropImg.src = d.crop_url || "";
  frameImg.src = d.frame_url || "";

  if (d.heatmap_url) {
    heatmapImg.src = d.heatmap_url;
    heatmapImg.style.display = "block";
    heatmapFallback.style.display = "none";
  } else {
    heatmapImg.style.display = "none";
    heatmapFallback.style.display = "block";
  }

  renderList();
}

// DECISION
async function sendDecision(decision) {
  if (!selected) return;

  try {
    await fetch("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: selected.id,
        decision
      })
    });

    if (decision === "reject") {
      detections = detections.filter(d => d.id !== selected.id);
    } else {
      detections = detections.map(d =>
        d.id === selected.id ? { ...d, status: "accepted" } : d
      );
    }

    selected = null;

    emptyState.style.display = "block";
    detailPanel.classList.add("hidden");

    renderList();
    updateStats();

  } catch (err) {
    console.error(err);
  }
}

acceptBtn.onclick = () => sendDecision("accept");
rejectBtn.onclick = () => sendDecision("reject");

// INIT LOOP
fetchData();
setInterval(fetchData, 2000);