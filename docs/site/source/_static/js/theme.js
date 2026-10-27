/* Keep Furo's controls and theme styling, with only light and dark states. */
(() => {
  function saveTheme(theme) {
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // Theme switching still works when browser storage is unavailable.
    }
  }

  function applyTheme(theme) {
    document.body.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    saveTheme(theme);

    const next = theme === "dark" ? "light" : "dark";
    for (const button of document.querySelectorAll("button.theme-toggle")) {
      button.setAttribute("aria-label", `Switch to ${next} theme`);
      button.setAttribute("aria-pressed", String(theme === "dark"));
    }
  }

  let stored;
  try {
    stored = localStorage.getItem("theme");
  } catch {
    // Use light mode when no saved preference can be read.
  }
  // Old "auto" preferences and first visits resolve to an explicit light state.
  applyTheme(stored === "dark" ? "dark" : "light");

  function bindControls() {
    for (const button of document.querySelectorAll("button.theme-toggle")) {
      button.querySelectorAll(".theme-icon-when-auto-light, .theme-icon-when-auto-dark")
        .forEach((icon) => icon.remove());
      button.addEventListener("click", (event) => {
        // Run before Furo's normal click handler, which includes an auto state.
        event.stopImmediatePropagation();
        applyTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
      }, { capture: true });
    }
    applyTheme(document.body.dataset.theme);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindControls, { once: true });
  } else {
    bindControls();
  }
})();
