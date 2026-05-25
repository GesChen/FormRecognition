/**
 * Web tools — appearance mode only.
 * Key: theme.mode ("dark"|"light")
 * Dispatches window "theme-changed" after apply (for canvas redraw).
 *
 * Accent is static and defined in tools/static/theme.css.
 */
(function () {
  var STORAGE_THEME = "theme.mode";
  var INSTANCE_KEY = "webtools.instance_id." + window.location.origin;
  var INSTANCE_POLL_MS = 4000;

  function root() {
    return document.documentElement;
  }

  function getTheme() {
    var t = localStorage.getItem(STORAGE_THEME);
    return t === "light" ? "light" : "dark";
  }

  function applyTheme(mode) {
    var normalized = mode === "light" ? "light" : "dark";
    localStorage.setItem(STORAGE_THEME, normalized);
    root().setAttribute("data-theme", normalized);
    window.dispatchEvent(new CustomEvent("theme-changed", { detail: { theme: normalized } }));
  }

  function nextTheme(mode) {
    return mode === "light" ? "dark" : "light";
  }

  function iconForTheme(mode) {
    return mode === "light" ? "\u263E" : "\u2600";
  }

  function labelForTheme(mode) {
    return mode === "light" ? "Switch to dark mode" : "Switch to light mode";
  }

  function injectFab() {
    if (document.getElementById("theme-fab")) return;

    var fab = document.createElement("div");
    fab.id = "theme-fab";
    fab.className = "theme-fab";
    fab.setAttribute("aria-label", "Theme settings");

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "theme-fab-toggle";
    toggle.setAttribute("aria-label", "Switch theme");

    function syncToggle() {
      var mode = getTheme();
      toggle.textContent = iconForTheme(mode);
      var next = nextTheme(mode);
      var label = labelForTheme(mode);
      toggle.title = label;
      toggle.setAttribute("aria-label", label);
      toggle.setAttribute("data-next-theme", next);
    }

    toggle.addEventListener("click", function () {
      applyTheme(nextTheme(getTheme()));
      syncToggle();
    });

    fab.appendChild(toggle);
    document.body.appendChild(fab);
    syncToggle();
  }

  function boot() {
    applyTheme(getTheme());
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () {
        injectFab();
        startInstanceGuard();
      });
    } else {
      injectFab();
      startInstanceGuard();
    }
  }

  function showExpiredOverlay() {
    if (!document.body) return;
    document.body.innerHTML =
      "<main style=\"font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 760px; margin: 64px auto; padding: 0 20px; color: #111;\">" +
      "<h2 style=\"margin-bottom: 8px;\">This tab is from an older server instance</h2>" +
      "<p style=\"margin-top: 0; line-height: 1.5;\">The web server was restarted. This tab is now expired to prevent stale writes. You can close this tab.</p>" +
      "<p style=\"line-height: 1.5;\">Open a fresh tab from the tools launcher to continue.</p>" +
      "</main>";
  }

  function expireThisTab() {
    try { window.close(); } catch (_) {}
    setTimeout(showExpiredOverlay, 80);
  }

  async function fetchInstanceId() {
    var res = await fetch("/api/instance?ts=" + Date.now(), {
      method: "GET",
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!res.ok) return null;
    var data = await res.json().catch(function () { return null; });
    if (!data || typeof data.instance_id !== "string" || !data.instance_id) return null;
    return data.instance_id;
  }

  function startInstanceGuard() {
    var stopped = false;
    window.addEventListener("beforeunload", function () { stopped = true; });

    async function check() {
      if (stopped) return;
      try {
        var current = await fetchInstanceId();
        if (!current) return;
        var prev = sessionStorage.getItem(INSTANCE_KEY);
        if (!prev) {
          sessionStorage.setItem(INSTANCE_KEY, current);
          return;
        }
        if (prev !== current) {
          sessionStorage.setItem(INSTANCE_KEY, current);
          expireThisTab();
          stopped = true;
        }
      } catch (_) {
        // Server may be briefly unavailable during restart; retry on next poll.
      }
    }

    check();
    setInterval(check, INSTANCE_POLL_MS);
  }

  boot();
})();
