import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <div className="nav">
      <div className="brand">⬡ DefectVision</div>

      <div className="nav-links">
        <NavLink to="/" className={({ isActive }) => isActive ? "active" : ""}>
          Live Feed
        </NavLink>

        <NavLink to="/dashboard" className={({ isActive }) => isActive ? "active" : ""}>
          Dashboard
        </NavLink>
      </div>
    </div>
  );
}