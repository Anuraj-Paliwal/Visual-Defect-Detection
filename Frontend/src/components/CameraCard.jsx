export default function CameraCard({ title, onClick }) {
  return (
    <div className="card" onClick={onClick}>
      {/* Optional preview */}
      <img
        src={`/video_feed?cam=${title.split(" ")[1]}`}
        alt={title}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover"
        }}
      />

      <div className="label">{title}</div>
      <div className="status">LIVE</div>
    </div>
  );
}