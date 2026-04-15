import { useState } from "react";
import CameraCard from "../components/CameraCard";

const cameras = [
  { id: 1, name: "Cam 1" },
  { id: 2, name: "Cam 2" },
  { id: 3, name: "Cam 3" },
  { id: 4, name: "Cam 4" }
];

export default function Live() {
  const [activeCam, setActiveCam] = useState(null);

  return (
    <div className="page">

      {/* GRID */}
      <div className="grid">
        {cameras.map(cam => (
          <CameraCard
            key={cam.id}
            title={cam.name}
            onClick={() => setActiveCam(cam)}
          />
        ))}
      </div>

      {/* FULLSCREEN MODAL */}
      {activeCam && (
        <div className="modal" onClick={() => setActiveCam(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>

            <span className="close" onClick={() => setActiveCam(null)}>×</span>

            <h3>{activeCam.name}</h3>

            <div className="video-box">
              <img
                src={`/video_feed?cam=${activeCam.id}`}
                alt="live"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "contain"
                }}
              />
            </div>

          </div>
        </div>
      )}

    </div>
  );
}