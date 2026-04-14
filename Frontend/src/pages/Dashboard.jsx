import Stat from "../components/Stat";

export default function Dashboard() {
  return (
    <div className="page">

      <div className="stats">
        <Stat label="Total Detections" value="124" />
        <Stat label="Active Cameras" value="3" />
        <Stat label="Alerts Today" value="8" />
      </div>

      <div className="dashboard">
        <h1>Analytics Dashboard</h1>
        <p>Charts & insights will appear here</p>
      </div>

    </div>
  );
}