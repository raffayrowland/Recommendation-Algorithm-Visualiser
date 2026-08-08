export async function getJson(path, params = {}, signal) {
  const url = new URL(path, window.location.href);

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }

  const response = await fetch(url, {
    signal,
    headers: { Accept: "application/json" },
  });

  if (response.ok) return response.json();

  let message = "The request could not be completed.";
  try {
    const body = await response.json();
    message = body.detail || message;
  } catch {
    // The default message also covers non-JSON server errors.
  }
  throw new Error(message);
}

export function normalizePoint(point) {
  return {
    track_id: String(point.track_id),
    track_name: point.track_name || "Untitled track",
    artist_name: point.artist_name || "Unknown artist",
    isrc: point.isrc || "",
    x: Number(point.x),
    y: Number(point.y),
    z: Number(point.z),
  };
}

export function hasPosition(point) {
  return Boolean(point) && [point.x, point.y, point.z].every(Number.isFinite);
}
