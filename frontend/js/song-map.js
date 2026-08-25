import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";

import { hasPosition } from "./api.js";

const COLORS = {
  point: new THREE.Color("#b8b8b4"),
  neighbour: new THREE.Color("#ffe600"),
  selected: new THREE.Color("#ffffff"),
};

const DEFAULT_CAMERA_POSITION = new THREE.Vector3(0, 1.5, 43);
const DEFAULT_CAMERA_TARGET = new THREE.Vector3(0, 0, 0);

export function createSongMap({ canvas, hoverLabel, hoverTitle, hoverArtist, onPointClick, onEmptyClick }) {
  const state = {
    basePoints: [],
    renderedPoints: [],
    pointsById: new Map(),
    neighbours: [],
    neighbourIds: new Set(),
    selected: null,
    hovered: null,
    transform: null,
    mode: "map",
    pointerDown: null,
    mouse: new THREE.Vector2(2, 2),
    lastPointer: { x: 0, y: 0 },
  };
  const movement = {
    forward: false,
    back: false,
    left: false,
    right: false,
    up: false,
    down: false,
  };

  let cameraTween = null;
  let hoverFrame = null;

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x050505, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  const camera = new THREE.PerspectiveCamera(43, 1, 0.02, 600);
  camera.position.copy(DEFAULT_CAMERA_POSITION);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;
  orbitControls.dampingFactor = 0.065;
  orbitControls.enableRotate = true;
  orbitControls.enablePan = false;
  orbitControls.rotateSpeed = 0.55;
  orbitControls.minDistance = 1.6;
  orbitControls.maxDistance = 110;
  orbitControls.zoomToCursor = true;

  const exploreControls = new PointerLockControls(camera, document.body);
  exploreControls.pointerSpeed = 0.72;
  exploreControls.addEventListener("lock", () => document.body.classList.add("is-pointer-locked"));
  exploreControls.addEventListener("unlock", () => document.body.classList.remove("is-pointer-locked"));

  const cloud = new THREE.Points(new THREE.BufferGeometry(), createPointMaterial());
  scene.add(cloud);

  const raycaster = new THREE.Raycaster();
  const clock = new THREE.Clock();

  function createPointMaterial() {
    return new THREE.ShaderMaterial({
      uniforms: { pixelRatio: { value: pixelRatio() } },
      vertexColors: true,
      transparent: true,
      depthTest: true,
      depthWrite: true,
      vertexShader: `
        uniform float pixelRatio;
        attribute float pointSize;
        attribute float pointOpacity;
        varying vec3 pointColor;
        varying float opacity;

        void main() {
          pointColor = color;
          opacity = pointOpacity;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = pointSize * pixelRatio;
        }
      `,
      fragmentShader: `
        varying vec3 pointColor;
        varying float opacity;

        void main() {
          if (distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;
          gl_FragColor = vec4(pointColor, opacity);
        }
      `,
    });
  }

  function pixelRatio() {
    return Math.min(window.devicePixelRatio, 1.75);
  }

  function createTransform(points) {
    const bounds = new THREE.Box3();
    for (const point of points) {
      bounds.expandByPoint(new THREE.Vector3(point.x, point.y, point.z));
    }
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const longestSide = Math.max(size.x, size.y, size.z, 0.001);
    return { center, scale: 28 / longestSide };
  }

  function positionFor(point) {
    if (!state.transform) return new THREE.Vector3();
    return new THREE.Vector3(
      (point.x - state.transform.center.x) * state.transform.scale,
      (point.y - state.transform.center.y) * state.transform.scale,
      (point.z - state.transform.center.z) * state.transform.scale,
    );
  }

  function rebuild() {
    const points = new Map(state.basePoints.map((point) => [point.track_id, point]));
    if (hasPosition(state.selected)) points.set(state.selected.track_id, state.selected);
    for (const point of state.neighbours) points.set(point.track_id, point);

    state.renderedPoints = [...points.values()];
    state.pointsById = points;

    const positions = new Float32Array(state.renderedPoints.length * 3);
    const colors = new Float32Array(state.renderedPoints.length * 3);
    const sizes = new Float32Array(state.renderedPoints.length);
    const opacities = new Float32Array(state.renderedPoints.length);

    state.renderedPoints.forEach((point, index) => {
      const position = positionFor(point);
      positions[index * 3] = position.x;
      positions[index * 3 + 1] = position.y;
      positions[index * 3 + 2] = position.z;
      sizes[index] = 3;
    });

    cloud.geometry.dispose();
    cloud.geometry = new THREE.BufferGeometry();
    cloud.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    cloud.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    cloud.geometry.setAttribute("pointSize", new THREE.BufferAttribute(sizes, 1));
    cloud.geometry.setAttribute("pointOpacity", new THREE.BufferAttribute(opacities, 1));
    cloud.geometry.computeBoundingSphere();
    updateAppearance();
  }

  function updateAppearance() {
    const colors = cloud.geometry.getAttribute("color");
    const sizes = cloud.geometry.getAttribute("pointSize");
    const opacities = cloud.geometry.getAttribute("pointOpacity");
    if (!colors || !sizes || !opacities) return;

    const baseSize = state.renderedPoints.length > 10000
      ? 2.8
      : state.renderedPoints.length > 5000 ? 3.2 : 4;

    state.renderedPoints.forEach((point, index) => {
      const isSelected = point.track_id === state.selected?.track_id;
      const isHovered = point.track_id === state.hovered?.track_id;
      const isNeighbour = state.neighbourIds.has(point.track_id);
      const color = isSelected ? COLORS.selected : isNeighbour ? COLORS.neighbour : COLORS.point;

      colors.setXYZ(index, color.r, color.g, color.b);
      sizes.setX(index, isSelected ? 6.5 : isHovered ? 6 : isNeighbour ? baseSize + 1 : baseSize);
      opacities.setX(index, state.selected && !isSelected && !isNeighbour ? 0.5 : 1);
    });

    colors.needsUpdate = true;
    sizes.needsUpdate = true;
    opacities.needsUpdate = true;
  }

  function setPoints(points) {
    state.basePoints = points;
    state.transform = createTransform(points);
    rebuild();
  }

  function setSelection(point, neighbours = []) {
    state.selected = point;
    state.neighbours = neighbours;
    state.neighbourIds = new Set(neighbours.map((item) => item.track_id));
    rebuild();
  }

  function clearSelection({ rebuildCloud = true } = {}) {
    state.selected = null;
    state.neighbours = [];
    state.neighbourIds.clear();
    hideHover();
    if (rebuildCloud) rebuild();
  }

  function findPoint(trackId) {
    return state.pointsById.get(String(trackId));
  }

  function focus(point) {
    const destination = positionFor(point);
    const direction = camera.position.clone().sub(orbitControls.target).normalize();
    if (!Number.isFinite(direction.x) || direction.lengthSq() < 0.1) direction.set(0, 0.1, 1);
    startCameraTween(destination.clone().add(direction.multiplyScalar(4.8)), destination, 850);
  }

  function resetCamera(animated = true) {
    if (animated) {
      startCameraTween(DEFAULT_CAMERA_POSITION, DEFAULT_CAMERA_TARGET, 900);
      return;
    }
    camera.position.copy(DEFAULT_CAMERA_POSITION);
    orbitControls.target.copy(DEFAULT_CAMERA_TARGET);
    camera.lookAt(DEFAULT_CAMERA_TARGET);
    orbitControls.update();
  }

  function startCameraTween(position, target, duration) {
    cameraTween = {
      start: performance.now(),
      duration,
      fromPosition: camera.position.clone(),
      toPosition: position.clone(),
      fromTarget: orbitControls.target.clone(),
      toTarget: target.clone(),
    };
  }

  function updateCameraTween(now) {
    if (!cameraTween) return;
    const progress = Math.min(1, (now - cameraTween.start) / cameraTween.duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    camera.position.lerpVectors(cameraTween.fromPosition, cameraTween.toPosition, eased);
    orbitControls.target.lerpVectors(cameraTween.fromTarget, cameraTween.toTarget, eased);
    camera.lookAt(orbitControls.target);
    if (progress >= 1) cameraTween = null;
  }

  function setMode(mode) {
    if (mode === state.mode) return;
    state.mode = mode;
    const isExplore = mode === "explore";
    document.body.classList.toggle("is-explore", isExplore);
    orbitControls.enabled = !isExplore;
    hideHover();

    if (!isExplore) {
      if (exploreControls.isLocked) exploreControls.unlock();
      orbitControls.target.copy(state.selected ? positionFor(state.selected) : DEFAULT_CAMERA_TARGET);
      orbitControls.update();
    }
  }

  function highlight(point) {
    if (!point) return;
    state.hovered = point;
    updateAppearance();
  }

  function clearHighlight() {
    if (!state.hovered) return;
    state.hovered = null;
    updateAppearance();
  }

  function handlePointerMove(event) {
    state.lastPointer = { x: event.clientX, y: event.clientY };
    state.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    state.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    if (state.mode !== "map" || !state.renderedPoints.length || hoverFrame) return;

    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = null;
      pickHoveredPoint();
    });
  }

  function pickHoveredPoint() {
    raycaster.params.Points.threshold = Math.max(
      0.16,
      camera.position.distanceTo(orbitControls.target) * 0.0065,
    );
    raycaster.setFromCamera(state.mouse, camera);
    const hit = raycaster.intersectObject(cloud, false)[0];
    const point = hit ? state.renderedPoints[hit.index] : null;

    if (point?.track_id === state.hovered?.track_id) {
      positionHoverLabel();
      return;
    }

    clearHighlight();
    if (!point) {
      hideHover();
      canvas.style.cursor = "grab";
      return;
    }

    highlight(point);
    canvas.style.cursor = "pointer";
    hoverTitle.textContent = point.track_name;
    hoverArtist.textContent = point.artist_name;
    hoverLabel.hidden = false;
    positionHoverLabel();
  }

  function positionHoverLabel() {
    const padding = 14;
    const width = hoverLabel.offsetWidth || 220;
    const height = hoverLabel.offsetHeight || 48;
    let { x, y } = state.lastPointer;

    if (x + width + 24 > window.innerWidth) x -= width + 28;
    if (y - height / 2 < padding) y = height / 2 + padding;
    if (y + height / 2 > window.innerHeight - padding) {
      y = window.innerHeight - height / 2 - padding;
    }

    hoverLabel.style.left = `${x}px`;
    hoverLabel.style.top = `${y}px`;
  }

  function hideHover() {
    hoverLabel.hidden = true;
    if (state.mode === "map") canvas.style.cursor = "grab";
    clearHighlight();
  }

  function lockExploreCamera() {
    try {
      const request = document.body.requestPointerLock();
      request?.catch?.(() => {});
    } catch {
      // A later canvas click can retry pointer lock.
    }
  }

  function setMovement(event, value) {
    const keyMap = {
      KeyW: "forward",
      ArrowUp: "forward",
      KeyS: "back",
      ArrowDown: "back",
      KeyA: "left",
      ArrowLeft: "left",
      KeyD: "right",
      ArrowRight: "right",
      Space: "up",
      ShiftLeft: "down",
      ShiftRight: "down",
    };
    const direction = keyMap[event.code];
    if (!direction || state.mode !== "explore") return;
    event.preventDefault();
    movement[direction] = value;
  }

  function updateExploreMovement(delta) {
    if (state.mode !== "explore" || !exploreControls.isLocked) return;
    const speed = Math.min(12, Math.max(2.5, camera.position.length() * 0.2));
    if (movement.forward) exploreControls.moveForward(speed * delta);
    if (movement.back) exploreControls.moveForward(-speed * delta);
    if (movement.left) exploreControls.moveRight(-speed * delta);
    if (movement.right) exploreControls.moveRight(speed * delta);
    if (movement.up) camera.position.y += speed * delta;
    if (movement.down) camera.position.y -= speed * delta;
  }

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(pixelRatio());
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    cloud.material.uniforms.pixelRatio.value = pixelRatio();
  }

  function animate(now = performance.now()) {
    requestAnimationFrame(animate);
    const delta = Math.min(clock.getDelta(), 0.05);
    updateCameraTween(now);
    if (state.mode === "map") orbitControls.update();
    else updateExploreMovement(delta);
    renderer.render(scene, camera);
  }

  window.addEventListener("resize", resize);
  window.addEventListener("keydown", (event) => setMovement(event, true));
  window.addEventListener("keyup", (event) => setMovement(event, false));
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerleave", hideHover);
  canvas.addEventListener("pointerdown", (event) => {
    state.pointerDown = { x: event.clientX, y: event.clientY };
    if (state.mode === "explore" && !exploreControls.isLocked) lockExploreCamera();
  });
  canvas.addEventListener("pointerup", (event) => {
    const start = state.pointerDown;
    state.pointerDown = null;
    if (!start || state.mode !== "map") return;
    const distance = Math.hypot(event.clientX - start.x, event.clientY - start.y);
    if (distance >= 5) return;
    if (state.hovered) onPointClick(state.hovered);
    else onEmptyClick();
  });

  resize();
  animate();

  return {
    clearHighlight,
    clearSelection,
    findPoint,
    focus,
    get mode() { return state.mode; },
    highlight,
    resetCamera,
    setMode,
    setPoints,
    setSelection,
  };
}
