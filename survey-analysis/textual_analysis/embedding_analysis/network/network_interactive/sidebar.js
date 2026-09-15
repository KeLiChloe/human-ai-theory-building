(function () {
  const TASK_ORDER = ["race", "gender"];
  const TASK_LABELS = { gender: "Gender", race: "Race" };
  const THEORY_ORDER = ["main-effects", "soi"];
  const THEORY_LABELS = {
    "main-effects": "Main Effects",
    "soi": "Second-Order Interactions",
  };
  const PHASE_ORDER = ["pre-ML", "post-ML"];
  const VARIANT_ORDER = ["three_groups", "collapsed"];
  const VARIANT_LABELS = {
    three_groups: "Three groups (PhDs, Senior Scientists, GenAI)",
    collapsed: "Two groups (Humans and GenAI)",
  };
  const OPEN_SECTIONS_KEY = "network-sidebar-open-sections";

  function navStorage() {
    try {
      return window.sessionStorage;
    } catch {
      return null;
    }
  }

  function readOpenSections() {
    const storage = navStorage();
    if (!storage) return new Set();
    try {
      const raw = storage.getItem(OPEN_SECTIONS_KEY);
      if (!raw) return new Set();
      const parsed = JSON.parse(raw);
      return new Set(Array.isArray(parsed) ? parsed : []);
    } catch {
      return new Set();
    }
  }

  function writeOpenSections(openIds) {
    const storage = navStorage();
    if (!storage) return;
    try {
      storage.setItem(OPEN_SECTIONS_KEY, JSON.stringify([...openIds]));
    } catch {
      /* ignore quota / private browsing */
    }
  }

  // Drop legacy persistence so reopening the site starts collapsed.
  try {
    localStorage.removeItem(OPEN_SECTIONS_KEY);
  } catch {
    /* ignore */
  }

  function sectionId(task, theory) {
    return task + "/" + theory;
  }

  const PAGE_SUFFIX_RE =
    /((?:gender|race)\/(?:main-effects|soi)\/(?:pre-ML|post-ML)\/[^/]+\/[^/]+?)(?:\.html)?$/i;

  function normalizeNavPath(path) {
    if (path == null) return "";
    let p = decodeURIComponent(String(path)).trim();
    if (p.includes("://")) {
      try {
        p = new URL(p).pathname;
      } catch {
        /* keep p */
      }
    }
    const pageMatch = p.match(PAGE_SUFFIX_RE);
    if (pageMatch) {
      return pageMatch[1].toLowerCase();
    }
    const marker = "/network/";
    const markerIndex = p.indexOf(marker);
    if (markerIndex !== -1) {
      p = p.slice(markerIndex + marker.length);
    }
    p = p.replace(/^\//, "").replace(/\/$/, "");
    if (!p || p === "index" || p === "index.html" || !p.includes("/")) return "";
    if (p.endsWith(".html")) p = p.slice(0, -5);
    if (p === "index") return "";
    return p.toLowerCase();
  }

  function resolvedPathFromHref(href) {
    try {
      return normalizeNavPath(new URL(href, window.location.href).pathname);
    } catch {
      return normalizeNavPath(href);
    }
  }

  function currentResolvedPath() {
    return normalizeNavPath(window.location.pathname);
  }

  function pathsMatch(itemHref) {
    const entryPath = normalizeNavPath(itemHref);
    const currentPath = currentResolvedPath();
    if (entryPath === currentPath) return true;
    try {
      return resolvedPathFromHref(navLinkHref(itemHref)) === currentPath;
    } catch {
      return false;
    }
  }

  /** Path to current page relative to the network site root (no leading slash). */
  function networkPathSuffix() {
    const path = decodeURIComponent(window.location.pathname);
    const pageMatch = path.match(
      /((?:gender|race)\/(?:main-effects|soi)\/(?:pre-ML|post-ML)\/[^/]+\/[^/]+\.html)$/i
    );
    if (pageMatch) return pageMatch[1];
    const marker = "/network/";
    const markerIndex = path.indexOf(marker);
    if (markerIndex !== -1) {
      const rest = path.slice(markerIndex + marker.length).replace(/\/$/, "");
      return rest || "index.html";
    }
    if (/\/index\.html$/i.test(path) || /\/$/.test(path)) return "index.html";
    return path.replace(/^\//, "").replace(/\/$/, "");
  }

  function currentNavPath() {
    return currentResolvedPath();
  }

  function isHttpDeployment() {
    const p = window.location.protocol;
    return p === "http:" || p === "https:";
  }

  /**
   * Host-absolute site root for http(s), e.g. "/human-ai-semantic-network" or "/network".
   * Relative links break when the URL omits a trailing slash
   * (/human-ai-semantic-network + "gender/..." => /gender/...).
   */
  function networkSiteRoot() {
    if (!isHttpDeployment()) return null;

    const scripts = document.getElementsByTagName("script");
    for (let i = 0; i < scripts.length; i++) {
      const src = scripts[i].src;
      if (!src || !/sidebar\.js(\?|$)/i.test(src)) continue;
      try {
        const pathname = new URL(src, window.location.href).pathname;
        return pathname.replace(/\/network_interactive\/sidebar\.js$/i, "") || "";
      } catch {
        /* keep looking */
      }
    }

    const path = decodeURIComponent(window.location.pathname);
    const underPage = path.match(
      /^(.*)\/(?:gender|race)\/(?:main-effects|soi)\/(?:pre-ML|post-ML)\//i
    );
    if (underPage) return underPage[1];

    let root = path.replace(/\/index\.html$/i, "").replace(/\/$/, "");
    return root || "";
  }

  /** Build href for a catalog entry from the current page location. */
  function navLinkHref(itemHref) {
    const target = String(itemHref).replace(/^\//, "");
    const root = networkSiteRoot();
    if (root !== null) {
      // Keep .html; project Pages need the real filename.
      return (root ? root : "") + "/" + target;
    }
    // file:// — stay relative to the current document.
    return navBasePrefix() + target;
  }

  function navBasePrefix() {
    const suffix = networkPathSuffix();
    if (!suffix || suffix === "index.html" || suffix === "index") return "";
    const depth = suffix.split("/").filter(Boolean).length - 1;
    return depth > 0 ? "../".repeat(depth) : "";
  }

  window.buildNetworkSidebar = function (entries, rootId, options) {
    const opts = options || {};
    const root = document.getElementById(rootId || "site-sidebar");
    if (!root) return;
    if (!entries || !entries.length) {
      console.warn("buildNetworkSidebar: no catalog entries; sidebar links will be empty.");
      entries = [];
    }

    const openSections = readOpenSections();

    const tree = {};
    entries.forEach((e) => {
      if (!tree[e.task]) tree[e.task] = {};
      if (!tree[e.task][e.theory_type]) tree[e.task][e.theory_type] = {};
      if (!tree[e.task][e.theory_type][e.phase]) tree[e.task][e.theory_type][e.phase] = {};
      tree[e.task][e.theory_type][e.phase][e.variant_key] = e;
    });

    const brand = document.createElement("div");
    brand.className = "sidebar-brand";
    const homeHref = navLinkHref("index.html");
    brand.innerHTML =
      '<a href="' + homeHref + '">' +
      '<div class="eyebrow">Theory explanation space</div>' +
      "<h1>Semantic Networks</h1></a>";
    root.appendChild(brand);

    const scroll = document.createElement("div");
    scroll.className = "sidebar-scroll";

    TASK_ORDER.forEach((task) => {
      if (!tree[task]) return;
      const taskEl = document.createElement("div");
      taskEl.className = "nav-task " + task;
      taskEl.innerHTML = '<h2 class="nav-task-title">' + TASK_LABELS[task] + "</h2>";
      let taskHasActive = false;

      THEORY_ORDER.forEach((theory) => {
        if (!tree[task][theory]) return;
        const details = document.createElement("details");
        details.className = "nav-theory";
        const summary = document.createElement("summary");
        summary.textContent = THEORY_LABELS[theory];
        details.appendChild(summary);
        let sectionHasActive = false;

        const sid = sectionId(task, theory);
        if (opts.expandAll || openSections.has(sid)) {
          details.open = true;
        }

        details.addEventListener("toggle", () => {
          if (details.open) openSections.add(sid);
          else openSections.delete(sid);
          writeOpenSections(openSections);
        });

        const theoryBody = document.createElement("div");
        theoryBody.className = "nav-theory-body";

        PHASE_ORDER.forEach((phase) => {
          const phaseEntries = tree[task][theory][phase];
          if (!phaseEntries) return;

          const phaseEl = document.createElement("div");
          phaseEl.className = "nav-phase";
          const phaseIsActive = VARIANT_ORDER.some((vk) => {
            const item = phaseEntries[vk];
            return item && pathsMatch(item.href);
          });
          if (phaseIsActive) phaseEl.classList.add("has-active");
          phaseEl.innerHTML = '<div class="nav-phase-label">' + phase + "</div>";

          const links = document.createElement("div");
          links.className = "nav-links";

          VARIANT_ORDER.forEach((vk) => {
            const item = phaseEntries[vk];
            if (!item) return;
            const a = document.createElement("a");
            a.href = navLinkHref(item.href);
            a.textContent = VARIANT_LABELS[vk];
            if (pathsMatch(item.href)) {
              a.classList.add("active");
              a.setAttribute("aria-current", "page");
              sectionHasActive = true;
              taskHasActive = true;
            }
            links.appendChild(a);
          });

          phaseEl.appendChild(links);
          theoryBody.appendChild(phaseEl);
        });

        if (sectionHasActive) {
          details.classList.add("has-active");
          details.open = true;
        }

        details.appendChild(theoryBody);
        taskEl.appendChild(details);
      });

      if (taskHasActive) taskEl.classList.add("has-active");
      scroll.appendChild(taskEl);
    });

    root.appendChild(scroll);

    const activeLink = root.querySelector(".nav-links a.active");
    if (activeLink) {
      requestAnimationFrame(() => {
        activeLink.scrollIntoView({ block: "nearest", behavior: "auto" });
      });
    }
  };
})();
