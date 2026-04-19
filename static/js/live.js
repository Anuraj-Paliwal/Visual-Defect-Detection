const cameras = [
  { id: 1, name: "Cam 1" },
  { id: 2, name: "Cam 2" },
  { id: 3, name: "Cam 3" },
  { id: 4, name: "Cam 4" }
];

// STATE
let cameraOn = false;
let activeCam = null;

// ELEMENTS
const grid = document.getElementById("cameraGrid");
const toggleBtn = document.getElementById("toggleBtn");

const cooldownInput = document.getElementById("cooldownInput");
const setCooldownBtn = document.getElementById("setCooldownBtn");

// MODAL
const modal = document.getElementById("modal");
const modalTitle = document.getElementById("modalTitle");
const videoFeed = document.getElementById("videoFeed");
const closeModal = document.getElementById("closeModal");

// ------------------ CAMERA API ------------------

function startCamera() {
  fetch("/api/camera/start", { method: "POST" });
  cameraOn = true;
  toggleBtn.innerText = "Stop Camera";
}

function stopCamera() {
  fetch("/api/camera/stop", { method: "POST" });
  cameraOn = false;
  activeCam = null;
  toggleBtn.innerText = "Start Camera";
  closeModalFn();
}

function toggleCamera() {
  cameraOn ? stopCamera() : startCamera();
}

// ------------------ COOLDOWN ------------------

function setCooldown() {
  const value = cooldownInput.value;

  fetch("/api/set_cooldown", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cooldown: value })
  });
}

// ------------------ GRID ------------------

function renderCameras() {
  grid.innerHTML = "";

  cameras.forEach(cam => {
    const div = document.createElement("div");
    div.className = "card";
    div.innerText = cam.name;

    div.onclick = () => {
      if (cameraOn) openCamera(cam);
    };

    grid.appendChild(div);
  });
}

// ------------------ MODAL ------------------

function openCamera(cam) {
  activeCam = cam;

  modal.classList.remove("hidden");

  modalTitle.innerText = cam.name;

  videoFeed.src = `/video_feed?cam=${cam.id}&t=${Date.now()}`;
}

function closeModalFn() {
  modal.classList.add("hidden");
  videoFeed.src = "";
}

// ------------------ EVENTS ------------------

toggleBtn.onclick = toggleCamera;
setCooldownBtn.onclick = setCooldown;

closeModal.onclick = closeModalFn;

modal.onclick = (e) => {
  if (e.target === modal) closeModalFn();
};

// ------------------ INIT ------------------

renderCameras();