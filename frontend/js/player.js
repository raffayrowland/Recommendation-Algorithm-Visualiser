import { getJson } from "./api.js";
import { elements as els, setButtonIcon } from "./dom.js";

export function createPlayer(showToast) {
  let previewController = null;
  let selectedTrackId = null;

  function pause() {
    els.previewAudio.pause();
    updatePlayButton(false);
  }

  function reset(point) {
    previewController?.abort();
    previewController = null;
    selectedTrackId = String(point.track_id);
    pause();

    els.previewAudio.removeAttribute("src");
    els.previewAudio.load();
    els.playerTitle.textContent = point.track_name;
    els.playerArtist.textContent = point.artist_name;
    els.coverArt.removeAttribute("src");
    els.coverArt.alt = "";
    els.playButton.disabled = true;
    els.previewProgress.disabled = true;
    els.previewProgress.max = 30;
    els.previewProgress.value = 0;
    els.elapsedTime.textContent = "0:00";
    els.durationTime.textContent = "0:30";
    updateProgressFill();
  }

  async function load(point) {
    if (!point.isrc) {
      markUnavailable();
      return;
    }

    previewController?.abort();
    const controller = new AbortController();
    previewController = controller;

    try {
      const payload = await getJson("/api/preview", { isrc: point.isrc }, controller.signal);
      if (previewController !== controller || selectedTrackId !== String(point.track_id)) return;
      populate(payload, point);
    } catch (error) {
      if (error.name !== "AbortError") markUnavailable();
    }
  }

  function populate(payload, fallback) {
    const title = payload.title || fallback.track_name;
    els.playerTitle.textContent = title;
    els.playerArtist.textContent = payload.artist || fallback.artist_name;

    if (payload.picture_link) {
      els.coverArt.src = payload.picture_link;
      els.coverArt.alt = `Cover art for ${title}`;
    }

    if (!payload.preview_link) {
      markUnavailable();
      return;
    }

    els.previewAudio.src = payload.preview_link;
    els.playButton.disabled = false;
    els.previewProgress.disabled = false;
  }

  function show() {
    els.player.hidden = false;
  }

  function hide() {
    previewController?.abort();
    previewController = null;
    selectedTrackId = null;
    els.player.hidden = true;
    pause();
  }

  function markUnavailable() {
    els.playButton.disabled = true;
    els.previewProgress.disabled = true;
    els.durationTime.textContent = "--:--";
  }

  function updatePlayButton(isPlaying) {
    setButtonIcon(
      els.playButton,
      isPlaying ? "pause" : "play",
      isPlaying ? "Pause preview" : "Play preview",
    );
  }

  function updateProgressFill() {
    const max = Number(els.previewProgress.max) || 30;
    const fill = (Number(els.previewProgress.value) / max) * 100;
    els.previewProgress.style.setProperty("--range-fill", `${fill}%`);
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const minutes = Math.floor(seconds / 60);
    const remainder = Math.floor(seconds % 60);
    return `${minutes}:${String(remainder).padStart(2, "0")}`;
  }

  els.playButton.addEventListener("click", () => {
    if (els.previewAudio.paused) {
      els.previewAudio.play().catch(() => showToast("The preview could not be played."));
    } else {
      els.previewAudio.pause();
    }
  });

  els.previewAudio.addEventListener("play", () => updatePlayButton(true));
  els.previewAudio.addEventListener("pause", () => updatePlayButton(false));
  els.previewAudio.addEventListener("ended", () => {
    els.previewAudio.currentTime = 0;
    updatePlayButton(false);
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
    const isMuted = els.previewAudio.muted;
    setButtonIcon(els.volumeButton, isMuted ? "volume-x" : "volume-2", isMuted ? "Unmute preview" : "Mute preview");
  });

  return { hide, load, markUnavailable, reset, show };
}
