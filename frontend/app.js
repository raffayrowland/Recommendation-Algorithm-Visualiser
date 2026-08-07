import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const els = {
  canvas: $("#songspace"),
  brand: $(".brand"),
  searchShell: $("#search-shell"),
  searchForm: $("#search-form"),
  searchInput: $("#search-input"),
  searchClear: $("#search-clear"),
  searchPopover: $("#search-popover"),
  searchResults: $("#search-results"),
  searchCount: $("#search-count"),
  moreResults: $("#more-results"),
  searchEnd: $("#search-end"),
  mapMode: $("#map-mode"),
  exploreMode: $("#explore-mode"),
  pointCount: $("#point-count"),
  pointRange: $("#point-range"),
  pointCountLabel: $("#point-count-label"),
  loadSpace: $("#load-space"),
  alphaButtons: $$("[data-alpha]"),
  sceneStatus: $("#scene-status-text"),
  hoverLabel: $("#hover-label"),
  neighbourPanel: $("#neighbour-panel"),
  neighbourList: $("#neighbour-list"),
  neighbourCount: $("#neighbour-count"),
  neighbourAlpha: $("#neighbour-alpha"),
  selectedTitle: $("#selected-title"),
  selectedArtist: $("#selected-artist"),
  closePanel: $("#close-panel"),
  player: $("#player"),
  playerTitle: $("#player-title"),
  playerArtist: $("#player-artist"),
  coverArt: $("#cover-art"),
  playButton: $("#play-button"),
  previewAudio: $("#preview-audio"),
  previewProgress: $("#preview-progress"),
  elapsedTime: $("#elapsed-time"),
  durationTime: $("#duration-time"),
  volumeButton: $("#volume-button"),
  toast: $("#toast"),
  loadState: $("#load-state"),
  loadTitle: $("#load-title"),
  loadDetail: $("#load-detail"),
  retryLoad: $("#retry-load"),
};

const COLORS = {
  point: new THREE.Color("#b8b8b4"),
  neighbour: new THREE.Color("#d8c66f"),
  selected: new THREE.Color("#ffffff"),
};

const state = {
  n: 1000,
  alpha: "050",
  draftAlpha: "050",
  basePoints: [],
  renderedPoints: [],
  pointsById: new Map(),
  neighbourIds: new Set(),
  neighbours: [],
  selected: null,
  hovered: null,
  coordinateTransform: null,
  cameraMode: "map",
  isLoadingSpace: false,
  searchQuery: "",
  searchItems: [],
  searchLimit: 10,
  searchDone: false,
  searchIndex: -1,
  searchController: null,
  selectionController: null,
  previewController: null,
  toastTimer: null,
  pointerDown: null,
  mouse: new THREE.Vector2(2, 2),
  lastPointer: { x: 0, y: 0 },
};

let renderer;
let scene;
let camera;
let cloud;
let mapControls;
let exploreControls;
let cameraTween = null;
let hoverFrame = null;
let clock;

function initScene() {
  renderer = new THREE.WebGLRenderer({
    canvas: els.canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.setClearColor(0x050505, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  camera = new THREE.PerspectiveCamera(43, window.innerWidth / window.innerHeight, 0.02, 600);
  camera.position.set(0, 1.5, 43);

  mapControls = new OrbitControls(camera, renderer.domElement);
  mapControls.enableDamping = true;
  mapControls.dampingFactor = 0.065;
  mapControls.enableRotate = true;
  mapControls.enablePan = false;
  mapControls.rotateSpeed = 0.55;
  mapControls.minDistance = 1.6;
  mapControls.maxDistance = 110;
  mapControls.zoomToCursor = true;

  exploreControls = new PointerLockControls(camera, document.body);
  exploreControls.pointerSpeed = 0.72;
  exploreControls.addEventListener("lock", () => document.body.classList.add("is-pointer-locked"));
  exploreControls.addEventListener("unlock", () => document.body.classList.remove("is-pointer-locked"));

  cloud = new THREE.Points(
    new THREE.BufferGeometry(),
    new THREE.ShaderMaterial({
      uniforms: {
        pixelRatio: { value: Math.min(window.devicePixelRatio, 1.75) },
      },
      vertexColors: true,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      vertexShader: `
        uniform float pixelRatio;
        attribute float pointSize;
        varying vec3 pointColor;

        void main() {
          pointColor = color;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = pointSize * pixelRatio;
        }
      `,
      fragmentShader: `
        varying vec3 pointColor;

        void main() {
          if (distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;
          gl_FragColor = vec4(pointColor, 1.0);
        }
      `,
    }),
  );
  scene.add(cloud);

  clock = new THREE.Clock();
  animate();
}

function normalizePoint(raw) {
  return {
    track_id: String(raw.track_id),
    track_name: raw.track_name || "Untitled track",
    artist_name: raw.artist_name || "Unknown artist",
    isrc: raw.isrc || "",
    x: Number(raw.x),
    y: Number(raw.y),
    z: Number(raw.z),
  };
}

function createCoordinateTransform(points) {
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(new THREE.Vector3(point.x, point.y, point.z));
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const longestSide = Math.max(size.x, size.y, size.z, 0.001);
  return { center, scale: 28 / longestSide };
}

function getPointPosition(point) {
  const transform = state.coordinateTransform;
  if (!transform) return new THREE.Vector3();
  return new THREE.Vector3(
    (point.x - transform.center.x) * transform.scale,
    (point.y - transform.center.y) * transform.scale,
    (point.z - transform.center.z) * transform.scale,
  );
}

function rebuildCloud() {
  const combined = new Map(state.basePoints.map((point) => [point.track_id, point]));
  for (const point of state.neighbours) combined.set(point.track_id, point);
  state.renderedPoints = [...combined.values()];
  state.pointsById = combined;

  const positions = new Float32Array(state.renderedPoints.length * 3);
  const colors = new Float32Array(state.renderedPoints.length * 3);
  const pointSizes = new Float32Array(state.renderedPoints.length);

  state.renderedPoints.forEach((point, index) => {
    const position = getPointPosition(point);
    positions[index * 3] = position.x;
    positions[index * 3 + 1] = position.y;
    positions[index * 3 + 2] = position.z;

    pointSizes[index] = 3;
  });

  cloud.geometry.dispose();
  cloud.geometry = new THREE.BufferGeometry();
  cloud.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  cloud.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  cloud.geometry.setAttribute("pointSize", new THREE.BufferAttribute(pointSizes, 1));
  cloud.geometry.computeBoundingSphere();
  updatePointAppearance();
}

function updatePointAppearance() {
  const colorAttribute = cloud.geometry.getAttribute("color");
  const sizeAttribute = cloud.geometry.getAttribute("pointSize");
  if (!colorAttribute || !sizeAttribute) return;

  const baseSize = state.renderedPoints.length > 10000
    ? 2.8
    : state.renderedPoints.length > 5000
      ? 3.2
      : 4;

  state.renderedPoints.forEach((point, index) => {
    const isSelected = point.track_id === state.selected?.track_id;
    const isHovered = point.track_id === state.hovered?.track_id;
    const isNeighbour = state.neighbourIds.has(point.track_id);
    const color = isSelected
      ? COLORS.selected
      : isNeighbour
        ? COLORS.neighbour
        : COLORS.point;

    colorAttribute.setXYZ(index, color.r, color.g, color.b);
    sizeAttribute.setX(index, isSelected ? 6.5 : isHovered ? 6 : isNeighbour ? baseSize + 1 : baseSize);
  });

  colorAttribute.needsUpdate = true;
  sizeAttribute.needsUpdate = true;
}

async function api(path, params, signal) {
  const url = new URL(path, window.location.href);
  Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));
  const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = "The request could not be completed.";
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // Preserve the fallback message for non-JSON server errors.
    }
    throw new Error(detail);
  }
  return response.json();
}

async function loadSpace() {
  if (state.isLoadingSpace) return;
  const n = clampPointCount(els.pointCount.value);
  const alpha = state.draftAlpha;
  state.isLoadingSpace = true;
  els.loadSpace.classList.add("is-loading");
  els.loadSpace.disabled = true;
  els.loadState.classList.remove("has-error");
  els.retryLoad.hidden = true;
  els.loadTitle.textContent = `Mapping ${formatNumber(n)} songs`;
  els.loadDetail.textContent = "This can take a moment for a new point cloud.";
  document.body.classList.remove("has-cloud");
  setSceneStatus(`Loading ${formatNumber(n)} songs`);
  deselectPoint({ animate: false, preserveCloud: true });

  try {
    const payload = await api("/api/space", { n, alpha });
    const points = Array.isArray(payload) ? payload.map(normalizePoint) : [];
    if (!points.length) throw new Error("The server returned an empty point cloud.");

    state.n = n;
    state.alpha = alpha;
    state.basePoints = points;
    state.neighbours = [];
    state.neighbourIds.clear();
    state.coordinateTransform = createCoordinateTransform(points);
    rebuildCloud();
    resetCamera(false);
    document.body.classList.add("has-cloud");
    els.pointCountLabel.textContent = `${formatNumber(points.length)} songs`;
    setSceneStatus(`${formatNumber(points.length)} points · alpha ${alphaLabel(alpha)}`);
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

function clampPointCount(value) {
  const parsed = Number.parseInt(value, 10);
  const finite = Number.isFinite(parsed) ? parsed : 1000;
  return Math.round(Math.min(20000, Math.max(100, finite)) / 100) * 100;
}

function syncPointControls(value) {
  const clamped = clampPointCount(value);
  els.pointCount.value = clamped;
  els.pointRange.value = clamped;
  const fill = ((clamped - 100) / (20000 - 100)) * 100;
  els.pointRange.style.setProperty("--range-fill", `${fill}%`);
  els.pointCountLabel.textContent = `${formatNumber(clamped)} songs`;
}

function setSceneStatus(message) {
  els.sceneStatus.textContent = message;
}

function alphaLabel(alpha) {
  return `${Number.parseInt(alpha, 10)}%`;
}

function formatNumber(number) {
  return new Intl.NumberFormat("en-GB").format(number);
}

function focusCamera(point) {
  const destination = getPointPosition(point);
  const direction = camera.position.clone().sub(mapControls.target).normalize();
  if (!Number.isFinite(direction.x) || direction.lengthSq() < 0.1) direction.set(0, 0.1, 1);
  const endPosition = destination.clone().add(direction.multiplyScalar(4.8));
  startCameraTween(endPosition, destination, 850);
}

function resetCamera(animated = true) {
  const endPosition = new THREE.Vector3(0, 1.5, 43);
  const endTarget = new THREE.Vector3(0, 0, 0);
  if (animated) startCameraTween(endPosition, endTarget, 900);
  else {
    camera.position.copy(endPosition);
    mapControls.target.copy(endTarget);
    camera.lookAt(endTarget);
    mapControls.update();
  }
}

function startCameraTween(endPosition, endTarget, duration) {
  cameraTween = {
    start: performance.now(),
    duration,
    fromPosition: camera.position.clone(),
    toPosition: endPosition.clone(),
    fromTarget: mapControls.target.clone(),
    toTarget: endTarget.clone(),
  };
}

function updateCameraTween(now) {
  if (!cameraTween) return;
  const progress = Math.min(1, (now - cameraTween.start) / cameraTween.duration);
  const eased = 1 - Math.pow(1 - progress, 3);
  camera.position.lerpVectors(cameraTween.fromPosition, cameraTween.toPosition, eased);
  mapControls.target.lerpVectors(cameraTween.fromTarget, cameraTween.toTarget, eased);
  camera.lookAt(mapControls.target);
  if (progress >= 1) cameraTween = null;
}

async function selectPoint(point) {
  if (!point || state.isLoadingSpace) return;
  state.selectionController?.abort();
  state.previewController?.abort();
  state.selectionController = new AbortController();
  state.previewController = new AbortController();

  state.selected = point;
  state.neighbours = [];
  state.neighbourIds.clear();
  document.body.classList.add("has-selection");
  els.neighbourPanel.hidden = false;
  els.player.hidden = false;
  els.selectedTitle.textContent = point.track_name;
  els.selectedArtist.textContent = point.artist_name;
  els.neighbourAlpha.textContent = `Alpha ${alphaLabel(state.alpha)}`;
  els.neighbourCount.textContent = "Finding tracks";
  renderNeighbourSkeletons();
  resetPlayer(point);
  rebuildCloud();
  focusCamera(point);
  closeSearch();
  setSceneStatus(`Focusing · ${point.track_name}`);

  const neighbourRequest = api(
    "/api/nn",
    { track_id: point.track_id, n: state.n, alpha: state.alpha },
    state.selectionController.signal,
  );
  const previewRequest = point.isrc
    ? api("/api/preview", { isrc: point.isrc }, state.previewController.signal)
    : Promise.reject(new Error("Preview metadata is unavailable for this song."));

  neighbourRequest
    .then((payload) => {
      if (state.selected?.track_id !== point.track_id) return;
      state.neighbours = (Array.isArray(payload) ? payload : []).map(normalizePoint);
      state.neighbourIds = new Set(state.neighbours.map((item) => item.track_id));
      rebuildCloud();
      renderNeighbours();
      els.neighbourCount.textContent = `${state.neighbours.length} tracks`;
      setSceneStatus(`${state.neighbours.length} nearest · alpha ${alphaLabel(state.alpha)}`);
    })
    .catch((error) => {
      if (error.name === "AbortError") return;
      renderNeighbourError(error.message);
      els.neighbourCount.textContent = "Unavailable";
      showToast(error.message);
    });

  previewRequest
    .then((payload) => {
      if (state.selected?.track_id !== point.track_id) return;
      populatePlayer(payload, point);
    })
    .catch((error) => {
      if (error.name === "AbortError") return;
      markPreviewUnavailable();
    });
}

function deselectPoint({ animate = true, preserveCloud = false } = {}) {
  if (!state.selected && !document.body.classList.contains("has-selection")) return;
  state.selectionController?.abort();
  state.previewController?.abort();
  state.selected = null;
  state.neighbours = [];
  state.neighbourIds.clear();
  document.body.classList.remove("has-selection");
  els.neighbourPanel.hidden = true;
  els.player.hidden = true;
  pausePreview();
  hideHover();
  if (!preserveCloud) rebuildCloud();
  if (animate) resetCamera(true);
  setSceneStatus(`${formatNumber(state.basePoints.length)} points · alpha ${alphaLabel(state.alpha)}`);
}

function renderNeighbourSkeletons() {
  els.neighbourList.replaceChildren(...Array.from({ length: 9 }, () => {
    const row = document.createElement("div");
    row.className = "neighbour-skeleton";
    row.innerHTML = '<span class="skeleton-number"></span><span class="skeleton-lines"><span></span><span></span></span>';
    return row;
  }));
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
    button.addEventListener("mouseenter", () => highlightPoint(point));
    button.addEventListener("mouseleave", clearPointHighlight);
    fragment.append(button);
  });
  els.neighbourList.replaceChildren(fragment);
  refreshIcons();
}

function renderNeighbourError(message) {
  const empty = document.createElement("div");
  empty.className = "inline-empty";
  const strong = document.createElement("strong");
  strong.textContent = "No nearby tracks to show";
  const detail = document.createElement("span");
  detail.textContent = message;
  empty.append(strong, detail);
  els.neighbourList.replaceChildren(empty);
}

function resetPlayer(point) {
  pausePreview();
  els.previewAudio.removeAttribute("src");
  els.previewAudio.load();
  els.playerTitle.textContent = point.track_name;
  els.playerArtist.textContent = point.artist_name;
  els.coverArt.removeAttribute("src");
  els.coverArt.alt = "";
  els.playButton.disabled = true;
  els.previewProgress.disabled = true;
  els.previewProgress.value = 0;
  updateProgressFill();
  els.elapsedTime.textContent = "0:00";
  els.durationTime.textContent = "0:30";
}

function populatePlayer(payload, fallback) {
  els.playerTitle.textContent = payload.title || fallback.track_name;
  els.playerArtist.textContent = payload.artist || fallback.artist_name;
  if (payload.picture_link) {
    els.coverArt.src = payload.picture_link;
    els.coverArt.alt = `Cover art for ${payload.title || fallback.track_name}`;
  }
  if (payload.preview_link) {
    els.previewAudio.src = payload.preview_link;
    els.playButton.disabled = false;
    els.previewProgress.disabled = false;
  } else {
    markPreviewUnavailable();
  }
}

function markPreviewUnavailable() {
  els.playButton.disabled = true;
  els.previewProgress.disabled = true;
  els.durationTime.textContent = "--:--";
}

function pausePreview() {
  els.previewAudio.pause();
  setPlayIcon(false);
}

function setPlayIcon(isPlaying) {
  els.playButton.setAttribute("aria-label", isPlaying ? "Pause preview" : "Play preview");
  els.playButton.setAttribute("title", isPlaying ? "Pause preview" : "Play preview");
  els.playButton.innerHTML = `<i data-lucide="${isPlaying ? "pause" : "play"}" aria-hidden="true"></i>`;
  refreshIcons();
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function updateProgressFill() {
  const max = Number(els.previewProgress.max) || 30;
  const fill = (Number(els.previewProgress.value) / max) * 100;
  els.previewProgress.style.setProperty("--range-fill", `${fill}%`);
}

function scheduleSearch() {
  window.clearTimeout(scheduleSearch.timer);
  const query = els.searchInput.value.trim();
  els.searchForm.classList.toggle("has-value", Boolean(query));
  if (query.length < 2) {
    state.searchController?.abort();
    closeSearch();
    return;
  }
  scheduleSearch.timer = window.setTimeout(() => runSearch(query, 10), 260);
}

async function runSearch(query, limit) {
  state.searchController?.abort();
  state.searchController = new AbortController();
  state.searchQuery = query;
  state.searchLimit = limit;
  state.searchIndex = -1;
  els.searchForm.classList.add("is-loading");
  els.searchPopover.hidden = false;
  els.searchInput.setAttribute("aria-expanded", "true");

  try {
    const payload = await api("/api/search/", { query, n: limit }, state.searchController.signal);
    if (state.searchQuery !== query) return;
    state.searchItems = Array.isArray(payload.songs) ? payload.songs : [];
    state.searchDone = state.searchItems.length < limit || limit >= 50;
    renderSearchResults();
  } catch (error) {
    if (error.name === "AbortError") return;
    state.searchItems = [];
    state.searchDone = true;
    renderSearchResults(error.message);
  } finally {
    if (state.searchQuery === query) els.searchForm.classList.remove("is-loading");
  }
}

function renderSearchResults(errorMessage = "") {
  const fragment = document.createDocumentFragment();
  state.searchItems.forEach((item, index) => {
    const trackId = String(item.track_id);
    const cloudPoint = state.pointsById.get(trackId);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-result";
    button.setAttribute("role", "option");
    button.dataset.index = index;
    button.dataset.trackId = trackId;
    button.innerHTML = `
      <span class="result-index">${String(index + 1).padStart(2, "0")}</span>
      <span class="result-copy"><strong></strong><span></span></span>
      ${cloudPoint ? '<span class="cloud-match" title="In current point cloud"></span>' : ""}
    `;
    $("strong", button).textContent = item.track_name || "Untitled track";
    $(".result-copy span", button).textContent = item.artist_name || "Unknown artist";
    button.addEventListener("click", () => activateSearchItem(index));
    fragment.append(button);
  });

  if (!state.searchItems.length) {
    const empty = document.createElement("div");
    empty.className = "inline-empty";
    const strong = document.createElement("strong");
    strong.textContent = errorMessage ? "Search unavailable" : "No matching songs";
    const detail = document.createElement("span");
    detail.textContent = errorMessage || "Try a different title or artist.";
    empty.append(strong, detail);
    fragment.append(empty);
  }

  els.searchResults.replaceChildren(fragment);
  els.searchCount.textContent = `${state.searchItems.length} found`;
  els.moreResults.hidden = state.searchDone || !state.searchItems.length;
  els.searchEnd.hidden = !state.searchDone || !state.searchItems.length;
}

function activateSearchItem(index) {
  const item = state.searchItems[index];
  if (!item) return;
  const point = state.pointsById.get(String(item.track_id));
  if (point) {
    selectPoint(point);
  } else {
    showToast("This song is outside the current cloud. Load more points to bring it into view.");
  }
}

function closeSearch() {
  state.searchIndex = -1;
  els.searchPopover.hidden = true;
  els.searchInput.setAttribute("aria-expanded", "false");
}

function clearSearch() {
  state.searchController?.abort();
  els.searchInput.value = "";
  els.searchForm.classList.remove("has-value", "is-loading");
  state.searchItems = [];
  state.searchQuery = "";
  closeSearch();
  els.searchInput.focus();
}

function moveSearchSelection(direction) {
  const items = $$(".search-result", els.searchResults);
  if (!items.length) return;
  state.searchIndex = (state.searchIndex + direction + items.length) % items.length;
  items.forEach((item, index) => item.classList.toggle("is-keyboard-active", index === state.searchIndex));
  items[state.searchIndex].scrollIntoView({ block: "nearest" });
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  els.toast.textContent = message;
  els.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    els.toast.hidden = true;
  }, 3600);
}

function setCameraMode(mode) {
  if (mode === state.cameraMode) return;
  state.cameraMode = mode;
  const isExplore = mode === "explore";
  document.body.classList.toggle("is-explore", isExplore);
  els.mapMode.classList.toggle("is-active", !isExplore);
  els.exploreMode.classList.toggle("is-active", isExplore);
  els.mapMode.setAttribute("aria-pressed", String(!isExplore));
  els.exploreMode.setAttribute("aria-pressed", String(isExplore));
  mapControls.enabled = !isExplore;
  hideHover();
  if (isExplore) {
    setSceneStatus("Explore camera active");
  } else {
    if (exploreControls.isLocked) exploreControls.unlock();
    mapControls.target.copy(
      state.selected ? getPointPosition(state.selected) : new THREE.Vector3(0, 0, 0),
    );
    mapControls.update();
    setSceneStatus(
      state.selected
        ? `${state.neighbours.length} nearest · alpha ${alphaLabel(state.alpha)}`
        : `${formatNumber(state.basePoints.length)} points · alpha ${alphaLabel(state.alpha)}`,
    );
  }
}

function lockExploreCamera() {
  try {
    const request = document.body.requestPointerLock();
    if (request?.catch) request.catch(() => {});
  } catch (_) {
    // Explore mode stays active and the next canvas click can retry the lock.
  }
}

const movement = { forward: false, back: false, left: false, right: false, up: false, down: false };

function updateExploreMovement(delta) {
  if (state.cameraMode !== "explore" || !exploreControls.isLocked) return;
  const speed = Math.min(12, Math.max(2.5, camera.position.length() * 0.2));
  if (movement.forward) exploreControls.moveForward(speed * delta);
  if (movement.back) exploreControls.moveForward(-speed * delta);
  if (movement.left) exploreControls.moveRight(-speed * delta);
  if (movement.right) exploreControls.moveRight(speed * delta);
  if (movement.up) camera.position.y += speed * delta;
  if (movement.down) camera.position.y -= speed * delta;
}

function highlightPoint(point) {
  if (!point) return;
  state.hovered = point;
  updatePointAppearance();
}

function clearPointHighlight() {
  if (!state.hovered) return;
  state.hovered = null;
  updatePointAppearance();
}

function onPointerMove(event) {
  state.lastPointer = { x: event.clientX, y: event.clientY };
  state.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  state.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  if (state.cameraMode !== "map" || !state.renderedPoints.length) return;
  if (hoverFrame) return;
  hoverFrame = requestAnimationFrame(() => {
    hoverFrame = null;
    pickHoveredPoint();
  });
}

function pickHoveredPoint() {
  const raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = Math.max(0.16, camera.position.distanceTo(mapControls.target) * 0.0065);
  raycaster.setFromCamera(state.mouse, camera);
  const hit = raycaster.intersectObject(cloud, false)[0];
  const point = hit ? state.renderedPoints[hit.index] : null;
  if (point?.track_id === state.hovered?.track_id) {
    positionHoverLabel();
    return;
  }
  clearPointHighlight();
  if (!point) {
    hideHover();
    els.canvas.style.cursor = "grab";
    return;
  }
  highlightPoint(point);
  els.canvas.style.cursor = "pointer";
  $("strong", els.hoverLabel).textContent = point.track_name;
  $("span", els.hoverLabel).textContent = point.artist_name;
  els.hoverLabel.hidden = false;
  positionHoverLabel();
}

function positionHoverLabel() {
  const pad = 14;
  const width = els.hoverLabel.offsetWidth || 220;
  const height = els.hoverLabel.offsetHeight || 48;
  let x = state.lastPointer.x;
  let y = state.lastPointer.y;
  if (x + width + 24 > window.innerWidth) x -= width + 28;
  if (y - height / 2 < pad) y = height / 2 + pad;
  if (y + height / 2 > window.innerHeight - pad) y = window.innerHeight - height / 2 - pad;
  els.hoverLabel.style.left = `${x}px`;
  els.hoverLabel.style.top = `${y}px`;
}

function hideHover() {
  els.hoverLabel.hidden = true;
  if (state.cameraMode === "map") els.canvas.style.cursor = "grab";
  clearPointHighlight();
}

function selectHoveredOrDeselect() {
  if (state.hovered) selectPoint(state.hovered);
  else if (state.selected) deselectPoint();
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.75 } });
}

function animate(now = performance.now()) {
  requestAnimationFrame(animate);
  const delta = Math.min(clock?.getDelta() || 0, 0.05);
  updateCameraTween(now);
  if (state.cameraMode === "map") mapControls.update();
  else updateExploreMovement(delta);
  renderer.render(scene, camera);
}

function bindEvents() {
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    cloud.material.uniforms.pixelRatio.value = Math.min(window.devicePixelRatio, 1.75);
  });

  els.canvas.addEventListener("pointerdown", (event) => {
    state.pointerDown = { x: event.clientX, y: event.clientY };
    if (state.cameraMode === "explore" && !exploreControls.isLocked) lockExploreCamera();
  });
  els.canvas.addEventListener("pointermove", onPointerMove);
  els.canvas.addEventListener("pointerleave", hideHover);
  els.canvas.addEventListener("pointerup", (event) => {
    if (!state.pointerDown || state.cameraMode !== "map") return;
    const distance = Math.hypot(event.clientX - state.pointerDown.x, event.clientY - state.pointerDown.y);
    if (distance < 5) selectHoveredOrDeselect();
    state.pointerDown = null;
  });

  els.brand.addEventListener("click", (event) => {
    event.preventDefault();
    deselectPoint();
    resetCamera(true);
  });

  els.pointRange.addEventListener("input", () => syncPointControls(els.pointRange.value));
  els.pointCount.addEventListener("change", () => syncPointControls(els.pointCount.value));
  els.pointCount.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      syncPointControls(els.pointCount.value);
      loadSpace();
    }
  });
  els.alphaButtons.forEach((button) => button.addEventListener("click", () => {
    state.draftAlpha = button.dataset.alpha;
    els.alphaButtons.forEach((candidate) => candidate.classList.toggle("is-active", candidate === button));
  }));
  els.loadSpace.addEventListener("click", loadSpace);
  els.retryLoad.addEventListener("click", loadSpace);

  els.searchInput.addEventListener("input", scheduleSearch);
  els.searchInput.addEventListener("focus", () => {
    if (state.searchItems.length || els.searchInput.value.trim().length >= 2) {
      els.searchPopover.hidden = false;
      els.searchInput.setAttribute("aria-expanded", "true");
    }
  });
  els.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSearchSelection(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSearchSelection(-1);
    } else if (event.key === "Enter" && state.searchIndex >= 0) {
      event.preventDefault();
      activateSearchItem(state.searchIndex);
    } else if (event.key === "Escape") {
      closeSearch();
      els.searchInput.blur();
    }
  });
  els.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = els.searchInput.value.trim();
    if (state.searchIndex >= 0) activateSearchItem(state.searchIndex);
    else if (query.length >= 2) runSearch(query, 10);
  });
  els.searchClear.addEventListener("click", clearSearch);
  els.moreResults.addEventListener("click", () => runSearch(state.searchQuery, Math.min(50, state.searchLimit + 10)));

  document.addEventListener("pointerdown", (event) => {
    if (!els.searchShell.contains(event.target)) closeSearch();
  });

  els.mapMode.addEventListener("click", () => setCameraMode("map"));
  els.exploreMode.addEventListener("click", () => setCameraMode("explore"));
  els.closePanel.addEventListener("click", () => deselectPoint());

  els.playButton.addEventListener("click", () => {
    if (els.previewAudio.paused) els.previewAudio.play().catch(() => showToast("The preview could not be played."));
    else els.previewAudio.pause();
  });
  els.previewAudio.addEventListener("play", () => setPlayIcon(true));
  els.previewAudio.addEventListener("pause", () => setPlayIcon(false));
  els.previewAudio.addEventListener("ended", () => {
    els.previewAudio.currentTime = 0;
    setPlayIcon(false);
  });
  els.previewAudio.addEventListener("loadedmetadata", () => {
    const duration = Number.isFinite(els.previewAudio.duration) ? els.previewAudio.duration : 30;
    els.previewProgress.max = duration;
    els.durationTime.textContent = formatTime(duration);
    updateProgressFill();
  });
  els.previewAudio.addEventListener("timeupdate", () => {
    els.previewProgress.value = els.previewAudio.currentTime;
    els.elapsedTime.textContent = formatTime(els.previewAudio.currentTime);
    updateProgressFill();
  });
  els.previewProgress.addEventListener("input", () => {
    els.previewAudio.currentTime = Number(els.previewProgress.value);
    updateProgressFill();
  });
  els.volumeButton.addEventListener("click", () => {
    els.previewAudio.muted = !els.previewAudio.muted;
    els.volumeButton.innerHTML = `<i data-lucide="${els.previewAudio.muted ? "volume-x" : "volume-2"}" aria-hidden="true"></i>`;
    els.volumeButton.setAttribute("aria-label", els.previewAudio.muted ? "Unmute preview" : "Mute preview");
    els.volumeButton.setAttribute("title", els.previewAudio.muted ? "Unmute preview" : "Mute preview");
    refreshIcons();
  });

  const setMovement = (event, value) => {
    const keyMap = {
      KeyW: "forward", ArrowUp: "forward", KeyS: "back", ArrowDown: "back",
      KeyA: "left", ArrowLeft: "left", KeyD: "right", ArrowRight: "right",
      Space: "up", ShiftLeft: "down", ShiftRight: "down",
    };
    const direction = keyMap[event.code];
    if (!direction || state.cameraMode !== "explore") return;
    event.preventDefault();
    movement[direction] = value;
  };
  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== els.searchInput) {
      event.preventDefault();
      els.searchInput.focus();
      return;
    }
    if (event.key === "Escape" && state.cameraMode === "map" && state.selected) deselectPoint();
    setMovement(event, true);
  });
  window.addEventListener("keyup", (event) => setMovement(event, false));
}

function boot() {
  initScene();
  bindEvents();
  syncPointControls(state.n);
  refreshIcons();
  window.addEventListener("load", refreshIcons, { once: true });
  loadSpace();
}

boot();
