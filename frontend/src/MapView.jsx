import React, { useEffect, useMemo } from "react";
import { CircleMarker, GeoJSON, MapContainer, TileLayer, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ROLE_VAR = { hub: "--shared", member: "--shared", private: "--private", none: "--none" };
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const num = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));

function FitBounds({ data }) {
  const map = useMap();
  useEffect(() => {
    if (data?.features?.length) map.fitBounds(L.geoJSON(data).getBounds(), { padding: [16, 16] });
  }, [data, map]);
  return null;
}

function centroid(geom) {
  const ring = geom.type === "Polygon" ? geom.coordinates[0] : geom.coordinates[0][0];
  const n = ring.length;
  return [ring.reduce((s, p) => s + p[1], 0) / n, ring.reduce((s, p) => s + p[0], 0) / n];
}

export default function MapView({ buildings, routes }) {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const hubs = useMemo(
    () => (buildings?.features || []).filter((f) => f.properties.role === "hub"), [buildings]);
  const style = (f) => ({
    color: css("--surface-1"), weight: 1,
    fillColor: css(ROLE_VAR[f.properties.role] || "--none"), fillOpacity: 0.85,
  });
  const onEach = (f, layer) => {
    const p = f.properties;
    layer.bindPopup(
      `<b>Building ${p.building_id}</b> · ${p.role}${p.unit !== "none" ? ` · unit ${p.unit}` : ""}<br/>` +
      `Roof ${num(p.area_m2, 0)} m² · footprint confidence ${num(p.confidence, 2)}<br/>` +
      `Tank ${num(p.storage_m3)} m³ · days demand met ${pct(p.reliability_days)}<br/>` +
      `P(fail) ${pct(p.failure_prob)} · ${num(p.litres_per_rupee)} L per ₹` +
      (p.survey_rank ? `<br/>Survey rank ${p.survey_rank}` : ""));
  };
  if (!buildings) return <div className="map card">Loading map…</div>;
  return (
    <>
      <div className="map">
        <MapContainer center={[13.05, 80.24]} zoom={16} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            className={dark ? "tiles-dark" : ""} maxZoom={19} />
          <GeoJSON key={`b-${buildings.features.length}`} data={buildings} style={style} onEachFeature={onEach} />
          {routes?.features?.length > 0 && (
            <GeoJSON key={`r-${routes.features.length}`} data={routes}
                     style={{ color: css("--shared"), weight: 2, opacity: 0.9 }} />
          )}
          {hubs.map((f) => (
            <CircleMarker key={f.properties.building_id} center={centroid(f.geometry)} radius={6}
                          pathOptions={{ color: css("--surface-1"), weight: 2, fillColor: css("--text-primary"), fillOpacity: 1 }}>
              <Tooltip>Hub tank {f.properties.unit} · {num(f.properties.storage_m3)} m³</Tooltip>
            </CircleMarker>
          ))}
          <FitBounds data={buildings} />
        </MapContainer>
      </div>
      <div className="legend">
        <span><span className="swatch" style={{ background: "var(--shared)" }} />Shared cluster (dot = hub tank, lines = street pipe routes)</span>
        <span><span className="swatch" style={{ background: "var(--private)" }} />Private tank</span>
        <span><span className="swatch" style={{ background: "var(--none)" }} />No system recommended</span>
      </div>
    </>
  );
}
