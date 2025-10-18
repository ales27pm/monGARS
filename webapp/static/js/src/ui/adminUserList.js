import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { resolveConfig } from "../config.js";
import { createAuthService } from "../services/auth.js";
import { createHttpService } from "../services/http.js";

const h = React.createElement;

function formatTimestamp(date) {
  if (!(date instanceof Date)) {
    return null;
  }
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(date);
  } catch (err) {
    console.warn("Unable to format timestamp", err);
    return date.toISOString();
  }
}

function UserListApp({ http }) {
  const [state, setState] = useState({
    users: [],
    loading: true,
    error: null,
    refreshedAt: null,
  });
  const [filter, setFilter] = useState("");

  const refresh = useCallback(
    async ({ silent = false, signal } = {}) => {
      if (!silent) {
        setState((prev) => ({
          ...prev,
          loading: true,
          error: null,
        }));
      }
      try {
        const users = await http.listUsers({ signal });
        if (signal?.aborted) {
          return;
        }
        setState({
          users,
          loading: false,
          error: null,
          refreshedAt: new Date(),
        });
      } catch (err) {
        if (signal?.aborted) {
          return;
        }
        const message =
          (err && err.message) ||
          "Impossible de récupérer la liste des utilisateurs.";
        setState((prev) => ({
          ...prev,
          loading: false,
          error: message,
        }));
      }
    },
    [http],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh({ signal: controller.signal });
    return () => {
      controller.abort();
    };
  }, [refresh]);

  const filteredUsers = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!query) {
      return state.users;
    }
    return state.users.filter((username) =>
      username.toLowerCase().includes(query),
    );
  }, [filter, state.users]);

  const totalCount = state.users.length;
  const visibleCount = filteredUsers.length;

  const rows =
    state.loading && !totalCount
      ? [
          h(
            "tr",
            { key: "loading" },
            h(
              "td",
              { colSpan: 2, className: "text-center text-muted py-4" },
              "Chargement de la liste des utilisateurs…",
            ),
          ),
        ]
      : filteredUsers.length
        ? filteredUsers.map((username, index) =>
            h(
              "tr",
              { key: username },
              h("td", { className: "text-muted" }, String(index + 1)),
              h("td", { className: "fw-semibold" }, username),
            ),
          )
        : [
            h(
              "tr",
              { key: "empty" },
              h(
                "td",
                { colSpan: 2, className: "text-center text-muted py-4" },
                totalCount
                  ? "Aucun utilisateur ne correspond à votre recherche."
                  : "Aucun utilisateur enregistré pour le moment.",
              ),
            ),
          ];

  const statusMessage = state.loading
    ? "Actualisation en cours"
    : state.error
      ? `Erreur: ${state.error}`
      : `Liste chargée (${visibleCount}/${totalCount})`;

  const formattedTimestamp = formatTimestamp(state.refreshedAt);

  return h(
    "div",
    { className: "admin-user-list" },
    h(
      "div",
      {
        className:
          "d-flex flex-column flex-md-row align-items-md-center justify-content-between gap-3 mb-4",
      },
      h(
        "div",
        { className: "flex-grow-1" },
        h("h1", { className: "h4 mb-1" }, "Utilisateurs enregistrés"),
        h(
          "p",
          { className: "text-muted mb-0" },
          "Consultez et filtrez les comptes disposant d'un accès à la plateforme.",
        ),
      ),
      h(
        "div",
        { className: "d-flex align-items-center gap-2" },
        h(
          "button",
          {
            type: "button",
            className: "btn btn-outline-primary",
            onClick: () => refresh(),
            disabled: state.loading,
          },
          state.loading ? "Actualisation…" : "Actualiser",
        ),
      ),
    ),
    h(
      "div",
      {
        className: "visually-hidden",
        "aria-live": "polite",
        "aria-atomic": "true",
      },
      statusMessage,
    ),
    state.error
      ? h(
          "div",
          {
            className: "alert alert-danger",
            role: "alert",
            "aria-live": "assertive",
          },
          state.error,
        )
      : null,
    h(
      "div",
      { className: "row g-3 align-items-end mb-3" },
      h(
        "div",
        { className: "col-12 col-md-6" },
        h(
          "label",
          { className: "form-label", htmlFor: "user-list-filter" },
          "Filtrer les utilisateurs",
        ),
        h("input", {
          id: "user-list-filter",
          className: "form-control",
          type: "search",
          placeholder: "Rechercher par nom d'utilisateur",
          value: filter,
          onChange: (event) => setFilter(event.target.value),
          autoComplete: "off",
          spellCheck: "false",
        }),
      ),
      h(
        "div",
        { className: "col-12 col-md-6" },
        h(
          "div",
          { className: "text-md-end text-muted small" },
          formattedTimestamp
            ? `Dernière mise à jour : ${formattedTimestamp}`
            : "Liste en cours de chargement…",
        ),
      ),
    ),
    h(
      "div",
      { className: "table-responsive" },
      h(
        "table",
        { className: "table table-striped align-middle" },
        h(
          "caption",
          { className: "visually-hidden" },
          "Liste des utilisateurs",
        ),
        h(
          "thead",
          null,
          h(
            "tr",
            null,
            h("th", { scope: "col", style: { width: "5rem" } }, "#"),
            h("th", { scope: "col" }, "Nom d'utilisateur"),
          ),
        ),
        h("tbody", null, ...rows),
      ),
    ),
  );
}

/**
 * Locate the mount point for the admin user list UI.
 * Templates that enable the React experience should declare a
 * `<div data-user-list-root></div>` element; the component will fall back to
 * other known roots to support legacy templates.
 */
function findUserListRoot(doc) {
  return (
    doc.getElementById("user-list-root") ||
    doc.querySelector("[data-user-list-root]") ||
    doc.getElementById("admin-root") ||
    doc.getElementById("app-root")
  );
}

export function renderUserListPage({
  doc = document,
  rawConfig = window.chatConfig || {},
} = {}) {
  const rootElement = findUserListRoot(doc);
  if (!rootElement) {
    console.warn("User list root element not found; skipping render.");
    return null;
  }
  const config = resolveConfig(rawConfig);
  if (!config.isAdmin) {
    console.warn("Admin privileges are required to render the user list UI.");
    return null;
  }
  const auth = createAuthService(config);
  const http = createHttpService({ config, auth });
  const root = createRoot(rootElement);
  root.render(h(UserListApp, { http }));
  return root;
}
