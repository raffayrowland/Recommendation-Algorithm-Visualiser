import { getJson, normalizePoint } from "./api.js";
import { $, $$, createEmptyState, elements as els } from "./dom.js";

const DEFAULT_LIMIT = 10;
const MAX_LIMIT = 50;

export function createSearch({ findPoint, onSelect }) {
  const state = {
    query: "",
    items: [],
    limit: DEFAULT_LIMIT,
    isDone: false,
    activeIndex: -1,
    controller: null,
  };
  let debounceTimer = null;

  function schedule() {
    window.clearTimeout(debounceTimer);
    const query = els.searchInput.value.trim();
    els.searchForm.classList.toggle("has-value", Boolean(query));

    if (query.length < 2) {
      state.controller?.abort();
      els.searchForm.classList.remove("is-loading");
      close();
      return;
    }

    debounceTimer = window.setTimeout(() => run(query, DEFAULT_LIMIT), 260);
  }

  async function run(query, limit) {
    state.controller?.abort();
    const controller = new AbortController();
    state.controller = controller;
    state.query = query;
    state.limit = limit;
    state.activeIndex = -1;
    els.searchForm.classList.add("is-loading");
    open();

    try {
      const payload = await getJson("/api/search/", { query, n: limit }, controller.signal);
      if (state.controller !== controller) return;
      state.items = Array.isArray(payload.songs) ? payload.songs : [];
      state.isDone = state.items.length < limit || limit >= MAX_LIMIT;
      render();
    } catch (error) {
      if (error.name === "AbortError") return;
      state.items = [];
      state.isDone = true;
      render(error.message);
    } finally {
      if (state.controller === controller) els.searchForm.classList.remove("is-loading");
    }
  }

  function render(errorMessage = "") {
    const fragment = document.createDocumentFragment();

    state.items.forEach((item, index) => {
      const trackId = String(item.track_id);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "search-result";
      button.setAttribute("role", "option");
      button.dataset.index = index;
      button.dataset.trackId = trackId;
      button.innerHTML = `
        <span class="result-index">${String(index + 1).padStart(2, "0")}</span>
        <span class="result-copy"><strong></strong><span></span></span>
        ${findPoint(trackId) ? '<span class="cloud-match" title="In current point cloud"></span>' : ""}
      `;
      $("strong", button).textContent = item.track_name || "Untitled track";
      $(".result-copy span", button).textContent = item.artist_name || "Unknown artist";
      button.addEventListener("click", () => activate(index));
      fragment.append(button);
    });

    if (!state.items.length) {
      fragment.append(createEmptyState(
        errorMessage ? "Search unavailable" : "No matching songs",
        errorMessage || "Try a different title or artist.",
      ));
    }

    els.searchResults.replaceChildren(fragment);
    els.searchCount.textContent = `${state.items.length} found`;
    els.moreResults.hidden = state.isDone || !state.items.length;
  }

  function activate(index) {
    const item = state.items[index];
    if (!item) return;
    onSelect(findPoint(String(item.track_id)) || normalizePoint(item));
  }

  function moveSelection(direction) {
    const items = $$(".search-result", els.searchResults);
    if (!items.length) return;
    state.activeIndex = (state.activeIndex + direction + items.length) % items.length;
    items.forEach((item, index) => {
      item.classList.toggle("is-keyboard-active", index === state.activeIndex);
    });
    items[state.activeIndex].scrollIntoView({ block: "nearest" });
  }

  function open() {
    els.searchPopover.hidden = false;
    els.searchInput.setAttribute("aria-expanded", "true");
  }

  function close() {
    state.activeIndex = -1;
    els.searchPopover.hidden = true;
    els.searchInput.setAttribute("aria-expanded", "false");
  }

  function clear() {
    window.clearTimeout(debounceTimer);
    state.controller?.abort();
    state.items = [];
    state.query = "";
    els.searchInput.value = "";
    els.searchForm.classList.remove("has-value", "is-loading");
    close();
    els.searchInput.focus();
  }

  els.searchInput.addEventListener("input", schedule);
  els.searchInput.addEventListener("focus", () => {
    if (state.items.length || els.searchInput.value.trim().length >= 2) open();
  });
  els.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Enter" && state.activeIndex >= 0) {
      event.preventDefault();
      activate(state.activeIndex);
    } else if (event.key === "Escape") {
      close();
      els.searchInput.blur();
    }
  });
  els.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = els.searchInput.value.trim();
    if (state.activeIndex >= 0) activate(state.activeIndex);
    else if (query.length >= 2) run(query, DEFAULT_LIMIT);
  });
  els.searchClear.addEventListener("click", clear);
  els.moreResults.addEventListener("click", () => run(state.query, Math.min(MAX_LIMIT, state.limit + DEFAULT_LIMIT)));
  document.addEventListener("pointerdown", (event) => {
    if (!els.searchShell.contains(event.target)) close();
  });

  return {
    close,
    focus: () => els.searchInput.focus(),
  };
}
