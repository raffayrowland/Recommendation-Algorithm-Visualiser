export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const byId = (id) => document.getElementById(id);

export const elements = {
  canvas: byId("songspace"),
  brand: $(".brand"),
  searchShell: byId("search-shell"),
  searchForm: byId("search-form"),
  searchInput: byId("search-input"),
  searchClear: byId("search-clear"),
  searchPopover: byId("search-popover"),
  searchResults: byId("search-results"),
  searchCount: byId("search-count"),
  moreResults: byId("more-results"),
  searchEnd: byId("search-end"),
  mapMode: byId("map-mode"),
  exploreMode: byId("explore-mode"),
  pointCount: byId("point-count"),
  pointRange: byId("point-range"),
  pointCountLabel: byId("point-count-label"),
  loadSpace: byId("load-space"),
  sceneStatus: byId("scene-status-text"),
  hoverLabel: byId("hover-label"),
  hoverTitle: $("#hover-label strong"),
  hoverArtist: $("#hover-label span"),
  neighbourPanel: byId("neighbour-panel"),
  neighbourList: byId("neighbour-list"),
  neighbourCount: byId("neighbour-count"),
  selectedTitle: byId("selected-title"),
  selectedArtist: byId("selected-artist"),
  closePanel: byId("close-panel"),
  player: byId("player"),
  playerTitle: byId("player-title"),
  playerArtist: byId("player-artist"),
  coverArt: byId("cover-art"),
  playButton: byId("play-button"),
  previewAudio: byId("preview-audio"),
  previewProgress: byId("preview-progress"),
  elapsedTime: byId("elapsed-time"),
  durationTime: byId("duration-time"),
  volumeButton: byId("volume-button"),
  toast: byId("toast"),
  loadState: byId("load-state"),
  loadTitle: byId("load-title"),
  loadDetail: byId("load-detail"),
  retryLoad: byId("retry-load"),
};

export function refreshIcons() {
  window.lucide?.createIcons({ attrs: { "stroke-width": 1.75 } });
}

export function setButtonIcon(button, icon, label) {
  button.innerHTML = `<i data-lucide="${icon}" aria-hidden="true"></i>`;
  if (label) {
    button.setAttribute("aria-label", label);
    button.setAttribute("title", label);
  }
  refreshIcons();
}

export function createEmptyState(title, detail) {
  const empty = document.createElement("div");
  const heading = document.createElement("strong");
  const message = document.createElement("span");

  empty.className = "inline-empty";
  heading.textContent = title;
  message.textContent = detail;
  empty.append(heading, message);
  return empty;
}
