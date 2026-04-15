import { useState } from "react";
import Stat from "../components/Stat";

export default function Dashboard() {
  const [selectedDefect, setSelectedDefect] = useState(null);

  const defects = [
    { id: 1, name: "Crack Detection" },
    { id: 2, name: "Surface Damage" },
    { id: 3, name: "Misalignment" },
    { id: 4, name: "Overheating" },
  ];

  return (
    <div className="page">

      {/* STATS */}
      <div className="stats">
        <Stat label="Total Detections" value="124" />
        <Stat label="Active Cameras" value="3" />
        <Stat label="Alerts Today" value="8" />
      </div>

      {/* MAIN DASHBOARD */}
      <div className="dashboard-layout">

        {/* LEFT SECTION */}
        <div className="dashboard-left">
          <h1>Analytics Dashboard</h1>
          <p>Charts & insights will appear here</p>
        </div>

        {/* RIGHT SECTION (DEFECT LIST) */}
        <div className="defect-panel">
          <h3>No. of Defects</h3>

          <ul>
            {defects.map((defect) => (
              <li
                key={defect.id}
                onClick={() => setSelectedDefect(defect)}
              >
                {defect.name}
              </li>
            ))}
          </ul>
        </div>

      </div>

      {/* MODAL */}
      {selectedDefect && (
        <div className="modal">
          <div className="modal-content two-col">

            <span
              className="close"
              onClick={() => setSelectedDefect(null)}
            >
              ×
            </span>

            {/* LEFT - HEATMAP */}
            <div className="heatmap-section">
              <h3>Heatmap</h3>

              <div className="heatmap-box">
                <img src="/heatmap.jpg" alt="heatmap" />
              </div>

              <div className="actions">
                <button className="accept">✔ Accept</button>
                <button className="reject">✖ Reject</button>
              </div>
            </div>

            {/* RIGHT - IMAGE */}
            <div className="image-section">
              <h3>Image</h3>

              <div className="image-box">
                <img src="/sample.jpg" alt="defect" />
              </div>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}