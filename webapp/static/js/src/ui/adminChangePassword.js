import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { resolveConfig } from "../config.js";
import { createAuthService } from "../services/auth.js";
import { createHttpService } from "../services/http.js";

const h = React.createElement;
const MIN_PASSWORD_LENGTH = 8;

function ChangePasswordApp({ http }) {
  const [form, setForm] = useState({
    oldPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [status, setStatus] = useState({
    submitting: false,
    error: null,
    success: null,
  });

  const passwordMismatch =
    form.newPassword !== "" &&
    form.confirmPassword !== "" &&
    form.newPassword !== form.confirmPassword;

  const meetsLengthRequirements =
    form.oldPassword.length >= MIN_PASSWORD_LENGTH &&
    form.newPassword.length >= MIN_PASSWORD_LENGTH &&
    form.confirmPassword.length >= MIN_PASSWORD_LENGTH;

  const canSubmit =
    meetsLengthRequirements && !passwordMismatch && !status.submitting;

  const liveMessage = status.submitting
    ? "Enregistrement du nouveau mot de passe"
    : status.error
      ? `Erreur: ${status.error}`
      : status.success
        ? status.success
        : "Formulaire prêt";

  const constraints = useMemo(() => {
    return [
      `Au moins ${MIN_PASSWORD_LENGTH} caractères`,
      "Doit différer du mot de passe actuel",
      "Confirmation identique au nouveau mot de passe",
    ];
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setStatus({ submitting: true, error: null, success: null });
    try {
      const payload = await http.changePassword({
        oldPassword: form.oldPassword,
        newPassword: form.newPassword,
      });
      const responseMessage =
        payload && typeof payload.status === "string"
          ? payload.status === "changed"
            ? "Mot de passe mis à jour avec succès."
            : `Mot de passe mis à jour (${payload.status}).`
          : "Mot de passe mis à jour avec succès.";
      setStatus({
        submitting: false,
        error: null,
        success: responseMessage,
      });
      setForm({ oldPassword: "", newPassword: "", confirmPassword: "" });
    } catch (err) {
      setStatus({
        submitting: false,
        error:
          (err && err.message) ||
          "Impossible de mettre à jour le mot de passe pour le moment.",
        success: null,
      });
    }
  }

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }

  return h(
    "form",
    {
      className: "change-password-form",
      onSubmit: handleSubmit,
      noValidate: true,
    },
    h(
      "div",
      { className: "mb-4" },
      h("h1", { className: "h4 mb-1" }, "Modifier le mot de passe"),
      h(
        "p",
        { className: "text-muted mb-0" },
        "Choisissez un mot de passe robuste pour protéger l'accès administrateur.",
      ),
    ),
    h(
      "div",
      {
        className: "visually-hidden",
        "aria-live": "polite",
        "aria-atomic": "true",
      },
      liveMessage,
    ),
    status.success
      ? h(
          "div",
          { className: "alert alert-success", role: "status" },
          status.success,
        )
      : null,
    status.error
      ? h(
          "div",
          { className: "alert alert-danger", role: "alert" },
          status.error,
        )
      : null,
    h(
      "div",
      { className: "mb-3" },
      h(
        "label",
        { className: "form-label", htmlFor: "old-password" },
        "Mot de passe actuel",
      ),
      h("input", {
        id: "old-password",
        name: "oldPassword",
        type: "password",
        className: "form-control",
        value: form.oldPassword,
        onChange: handleChange,
        autoComplete: "current-password",
        minLength: MIN_PASSWORD_LENGTH,
        required: true,
      }),
    ),
    h(
      "div",
      { className: "mb-3" },
      h(
        "label",
        { className: "form-label", htmlFor: "new-password" },
        "Nouveau mot de passe",
      ),
      h("input", {
        id: "new-password",
        name: "newPassword",
        type: "password",
        className: "form-control",
        value: form.newPassword,
        onChange: handleChange,
        autoComplete: "new-password",
        minLength: MIN_PASSWORD_LENGTH,
        required: true,
      }),
    ),
    h(
      "div",
      { className: "mb-3" },
      h(
        "label",
        { className: "form-label", htmlFor: "confirm-password" },
        "Confirmer le nouveau mot de passe",
      ),
      h("input", {
        id: "confirm-password",
        name: "confirmPassword",
        type: "password",
        className: "form-control",
        value: form.confirmPassword,
        onChange: handleChange,
        autoComplete: "new-password",
        minLength: MIN_PASSWORD_LENGTH,
        required: true,
        "aria-invalid": passwordMismatch ? "true" : "false",
        "aria-describedby": passwordMismatch ? "password-mismatch" : undefined,
      }),
      passwordMismatch
        ? h(
            "div",
            {
              id: "password-mismatch",
              className: "text-danger small mt-2",
            },
            "Le nouveau mot de passe et sa confirmation doivent correspondre.",
          )
        : null,
    ),
    h(
      "div",
      { className: "mb-4" },
      h(
        "p",
        { className: "text-muted small mb-2" },
        "Recommandations de sécurité :",
      ),
      h(
        "ul",
        { className: "small text-muted ps-3 mb-0" },
        ...constraints.map((rule) => h("li", { key: rule }, rule)),
      ),
    ),
    h(
      "button",
      {
        type: "submit",
        className: "btn btn-primary",
        disabled: !canSubmit,
      },
      status.submitting ? "Enregistrement…" : "Mettre à jour le mot de passe",
    ),
  );
}

function findChangePasswordRoot(doc) {
  return (
    doc.getElementById("change-password-root") ||
    doc.querySelector("[data-change-password-root]") ||
    doc.getElementById("admin-root") ||
    doc.getElementById("app-root")
  );
}

export function renderChangePasswordPage({
  doc = document,
  rawConfig = window.chatConfig || {},
} = {}) {
  const rootElement = findChangePasswordRoot(doc);
  if (!rootElement) {
    console.warn("Change-password root element not found; skipping render.");
    return null;
  }
  const config = resolveConfig(rawConfig);
  const auth = createAuthService(config);
  const http = createHttpService({ config, auth });
  const root = createRoot(rootElement);
  root.render(h(ChangePasswordApp, { http }));
  return root;
}
