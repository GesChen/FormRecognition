# EVMS web tools — theme

All tool pages under `tools/templates/` load shared theme assets:

- `tools/static/evms-theme.css` — dark/light tokens, legacy CSS variable aliases (`--bg`, `--accent`, …), floating settings panel styles.
- `tools/static/evms-theme.js` — reads preferences, sets `html[data-evms-theme]`, injects the gear control (bottom-right).

## User preferences (browser)

Stored in `localStorage`:

| Key | Values | Notes |
|-----|--------|--------|
| `evms.theme` | `dark` (default) or `light` | Applied on load. |
| `evms.accent` | `#RRGGBB` | Optional; if unset, CSS defaults per theme apply. |

Changing appearance or accent dispatches `window` event **`evms-theme-changed`** (detail may include `theme` or `accent`). Canvas-based pages listen and redraw.

## Changing defaults (project)

Edit **`tools/static/evms-theme.css`**:

- **`html[data-evms-theme="dark"]`** / **`"light"`** — backgrounds, borders, syntax colors for pipeline debug, default `--evms-accent`, etc.
- Legacy block **`html[data-evms-theme]`** maps `--evms-*` to names older templates use (`--panel`, `--accent-hover`, `--input-bg`, …).

Default `data-evms-theme="dark"` on `<html>` avoids a flash before the script runs; the script immediately syncs from `localStorage`.
