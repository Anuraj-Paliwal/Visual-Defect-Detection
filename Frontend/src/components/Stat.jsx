export default function Stat({ label, value }) {
  return (
    <div className="stat">
      <p>{label}</p>
      <h2>{value}</h2>
    </div>
  );
}