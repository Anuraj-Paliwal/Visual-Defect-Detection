// ── Stats ──────────────────────────────────────────────────────────────────
let barChart = null;

async function loadStats() {
  const res   = await fetch("/api/stats");
  const stats = await res.json();

  const total = stats.reduce((s, r) => s + r.count, 0);
  document.getElementById("s-total").textContent   = total.toLocaleString();
  document.getElementById("s-classes").textContent = stats.length;

  const top = stats.sort((a, b) => b.count - a.count)[0];
  document.getElementById("s-top").textContent = top ? top.class : "—";

  // Bar chart
  const labels = stats.map(r => r.class);
  const counts = stats.map(r => r.count);
  const colors = labels.map((_, i) =>
    ["#3b82f6","#22c55e","#f59e0b","#ef4444","#a855f7","#06b6d4"][i % 6]);

  if (barChart) { barChart.destroy(); }
  const ctx = document.getElementById("bar-chart").getContext("2d");
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Count", data: counts, backgroundColor: colors, borderRadius: 6, borderSkipped: false }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#2e3250" }, ticks: { color: "#7b8099" } },
        y: { grid: { color: "#2e3250" }, ticks: { color: "#7b8099", stepSize: 1 } }
      }
    }
  });

  // Populate class filter
  const select = document.getElementById("class-filter");
  const existing = Array.from(select.options).map(o => o.value);
  stats.forEach(r => {
    if (!existing.includes(r.class)) {
      const opt = document.createElement("option");
      opt.value = r.class;
      opt.textContent = r.class;
      select.appendChild(opt);
    }
  });
}

// ── Detection table ────────────────────────────────────────────────────────
async function loadRecent(cls = "") {
  const url = "/api/detections" + (cls ? `?class=${encodeURIComponent(cls)}` : "");
  const res  = await fetch(url);
  const rows = await res.json();

  document.getElementById("dash-body").innerHTML = rows.map(d => `
    <tr>
      <td>${d.timestamp}</td>
      <td>${d.class_name}</td>
      <td>${(d.confidence * 100).toFixed(1)}%</td>
      <td><span class="badge badge-${d.source === "webcam" ? "green" : "amber"}">${d.source}</span></td>
    </tr>`).join("");

  // Gallery
  const gallery = document.getElementById("gallery");
  gallery.innerHTML = rows.filter(d => d.crop_url).slice(0, 30).map(d => `
    <div class="gallery-item">
      <img src="${d.crop_url}" alt="${d.class_name}" loading="lazy"/>
      <div class="gallery-label">${d.class_name} ${(d.confidence * 100).toFixed(0)}%</div>
    </div>`).join("");
}

document.getElementById("class-filter").addEventListener("change", e => loadRecent(e.target.value));

// ── Init ───────────────────────────────────────────────────────────────────
loadStats();
loadRecent();
setInterval(() => { loadStats(); loadRecent(document.getElementById("class-filter").value); }, 10000);
