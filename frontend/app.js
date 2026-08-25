import { getJson, hasPosition, normalizePoint } from "./js/api.js";
import { $, createEmptyState, elements as els, refreshIcons } from "./js/dom.js";
import { createPlayer } from "./js/player.js";
import { createSearch } from "./js/search.js";
import { createSongMap } from "./js/song-map.js";

const state = {
  pointCount: 10000,
  loadedPointCount: 0,
  neighbours: [],
  selected: null,
  isLoadingSpace: false,
  selectionController: null,
  toastTimer: null,
};

const songMap = createSongMap({
  canvas: els.canvas,
  hoverLabel: els.hoverLabel,
  hoverTitle: els.hoverTitle,
  hoverArtist: els.hoverArtist,
  onPointClick: selectPoint,
  onEmptyClick: () => {
    if (state.selected) deselectPoint();
  },
});

const player = createPlayer(showToast);
const search = createSearch({
  findPoint: (trackId) => songMap.findPoint(trackId),
  onSelect: selectPoint,
});

async function loadSpace() {
  if (state.isLoadingSpace) return;

  const pointCount = clampPointCount(els.pointCount.value);
  state.isLoadingSpace = true;

  els.loadSpace.classList.add("is-loading");
  els.loadSpace.disabled = true;
  els.loadState.classList.remove("has-error");
  els.retryLoad.hidden = true;
  els.loadTitle.textContent = `Mapping ${formatNumber(pointCount)} songs`;
  els.loadDetail.textContent = "This can take a moment for a new point cloud.";
  document.body.classList.remove("has-cloud");
  setSceneStatus(`Loading ${formatNumber(pointCount)} songs`);
  deselectPoint({ animate: false, preserveCloud: true });

  try {
    const payload = await getJson("/api/space", { n: pointCount });
    const points = Array.isArray(payload) ? payload.map(normalizePoint) : [];
    if (!points.length) throw new Error("The server returned an empty point cloud.");

    state.pointCount = pointCount;
    state.loadedPointCount = points.length;
    state.neighbours = [];
    songMap.setPoints(points);
    songMap.resetCamera(false);

    document.body.classList.add("has-cloud");
    els.pointCountLabel.textContent = `${formatNumber(points.length)} songs`;
    setSceneStatus(`${formatNumber(points.length)} points`);
  } catch (error) {
    els.loadState.classList.add("has-error");
    els.loadTitle.textContent = "Songspace could not be loaded";
    els.loadDetail.textContent = error.message;
    els.retryLoad.hidden = false;
    setSceneStatus("Point cloud unavailable");
  } finally {
    state.isLoadingSpace = false;
    els.loadSpace.classList.remove("is-loading");
    els.loadSpace.disabled = false;
  }
}

async function selectPoint(point) {
  if (!point || state.isLoadingSpace) return;

  state.selectionController?.abort();
  const controller = new AbortController();
  state.selectionController = controller;
  const trackId = String(point.track_id);
  const wasPositioned = hasPosition(point);

  state.selected = point;
  state.neighbours = [];
  document.body.classList.add("has-selection");
  els.neighbourPanel.hidden = false;
  els.selectedTitle.textContent = point.track_name;
  els.selectedArtist.textContent = point.artist_name;
  els.neighbourCount.textContent = "Finding tracks";
  renderNeighbourSkeletons();

  player.show();
  player.reset(point);
  if (point.isrc) player.load(point);

  songMap.setSelection(point);
  if (wasPositioned) songMap.focus(point);
  search.close();
  setSceneStatus(wasPositioned ? `Focusing · ${point.track_name}` : `Locating · ${point.track_name}`);

  try {
    const payload = await getJson(
      "/api/nn",
      { track_id: trackId, n: state.pointCount },
      controller.signal,
    );
    if (state.selectionController !== controller || state.selected?.track_id !== trackId) return;

    const returnedPoints = (Array.isArray(payload) ? payload : []).map(normalizePoint);
    const positionedSelection = returnedPoints.find((item) => item.track_id === trackId);
    if (!wasPositioned && !positionedSelection) {
      throw new Error("The selected song could not be positioned in the current cloud.");
    }

    const selectedPoint = positionedSelection || point;
    state.selected = selectedPoint;
    state.neighbours = returnedPoints.filter((item) => item.track_id !== trackId);
    els.selectedTitle.textContent = selectedPoint.track_name;
    els.selectedArtist.textContent = selectedPoint.artist_name;
    els.neighbourCount.textContent = `${state.neighbours.length} tracks`;

    songMap.setSelection(selectedPoint, state.neighbours);
    if (!wasPositioned) songMap.focus(selectedPoint);
    renderNeighbours();
    setSceneStatus(`${state.neighbours.length} nearest`);

    if (!point.isrc) {
      player.reset(selectedPoint);
      player.load(selectedPoint);
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    renderNeighbourError(error.message);
    els.neighbourCount.textContent = "Unavailable";
    player.markUnavailable();
    showToast(error.message);
  }
}

function deselectPoint({ animate = true, preserveCloud = false } = {}) {
  if (!state.selected && !document.body.classList.contains("has-selection")) return;

  state.selectionController?.abort();
  state.selectionController = null;
  state.selected = null;
  state.neighbours = [];
  document.body.classList.remove("has-selection");
  els.neighbourPanel.hidden = true;
  player.hide();
  songMap.clearSelection({ rebuildCloud: !preserveCloud });
  if (animate) songMap.resetCamera(true);
  setSceneStatus(`${formatNumber(state.loadedPointCount)} points`);
}

function renderNeighbourSkeletons() {
  const rows = Array.from({ length: 9 }, () => {
    const row = document.createElement("div");
    row.className = "neighbour-skeleton";
    row.innerHTML = '<span class="skeleton-number"></span><span class="skeleton-lines"><span></span><span></span></span>';
    return row;
  });
  els.neighbourList.replaceChildren(...rows);
}

function renderNeighbours() {
  if (!state.neighbours.length) {
    renderNeighbourError("No neighbours were returned for this track.");
    return;
  }

  const fragment = document.createDocumentFragment();
  state.neighbours.forEach((point, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "neighbour-row";
    button.dataset.trackId = point.track_id;
    button.innerHTML = `
      <span class="neighbour-rank">${String(index + 1).padStart(2, "0")}</span>
      <span class="neighbour-copy"><strong></strong><span></span></span>
      <i data-lucide="arrow-up-right" aria-hidden="true"></i>
    `;
    $("strong", button).textContent = point.track_name;
    $(".neighbour-copy span", button).textContent = point.artist_name;
    button.addEventListener("click", () => selectPoint(point));
    button.addEventListener("mouseenter", () => songMap.highlight(point));
    button.addEventListener("mouseleave", songMap.clearHighlight);
    fragment.append(button);
  });

  els.neighbourList.replaceChildren(fragment);
  refreshIcons();
}

function renderNeighbourError(message) {
  els.neighbourList.replaceChildren(createEmptyState("No nearby tracks to show", message));
}

function clampPointCount(value) {
  const parsed = Number.parseInt(value, 10);
  const finiteValue = Number.isFinite(parsed) ? parsed : 1000;
  return Math.round(Math.min(20000, Math.max(100, finiteValue)) / 100) * 100;
}

function syncPointControls(value) {
  const pointCount = clampPointCount(value);
  const fill = ((pointCount - 100) / (20000 - 100)) * 100;
  els.pointCount.value = pointCount;
  els.pointRange.value = pointCount;
  els.pointRange.style.setProperty("--range-fill", `${fill}%`);
  els.pointCountLabel.textContent = `${formatNumber(pointCount)} songs`;
}

function setCameraMode(mode) {
  if (mode === songMap.mode) return;

  const isExplore = mode === "explore";
  songMap.setMode(mode);
  els.mapMode.classList.toggle("is-active", !isExplore);
  els.exploreMode.classList.toggle("is-active", isExplore);
  els.mapMode.setAttribute("aria-pressed", String(!isExplore));
  els.exploreMode.setAttribute("aria-pressed", String(isExplore));

  if (isExplore) {
    setSceneStatus("Explore camera active");
  } else if (state.selected) {
    setSceneStatus(`${state.neighbours.length} nearest`);
  } else {
    setSceneStatus(`${formatNumber(state.loadedPointCount)} points`);
  }
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  els.toast.textContent = message;
  els.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    els.toast.hidden = true;
  }, 3600);
}

function setSceneStatus(message) {
  els.sceneStatus.textContent = message;
}

function formatNumber(number) {
  return new Intl.NumberFormat("en-GB").format(number);
}

function bindControls() {
  els.brand.addEventListener("click", (event) => {
    event.preventDefault();
    const hadSelection = Boolean(state.selected);
    deselectPoint();
    if (!hadSelection) songMap.resetCamera(true);
  });

  els.pointRange.addEventListener("input", () => syncPointControls(els.pointRange.value));
  els.pointCount.addEventListener("change", () => syncPointControls(els.pointCount.value));
  els.pointCount.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    syncPointControls(els.pointCount.value);
    loadSpace();
  });

  els.loadSpace.addEventListener("click", loadSpace);
  els.retryLoad.addEventListener("click", loadSpace);
  els.mapMode.addEventListener("click", () => setCameraMode("map"));
  els.exploreMode.addEventListener("click", () => setCameraMode("explore"));
  els.closePanel.addEventListener("click", () => deselectPoint());

  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== els.searchInput) {
      event.preventDefault();
      search.focus();
    } else if (event.key === "Escape" && songMap.mode === "map" && state.selected) {
      deselectPoint();
    }
  });
}

function boot() {
  bindControls();
  syncPointControls(state.pointCount);
  refreshIcons();
  window.addEventListener("load", refreshIcons, { once: true });
  loadSpace();
}

boot();
