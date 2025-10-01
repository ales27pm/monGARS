export function createAuthService(config) {
  function persistToken(token) {
    if (!token) return;
// webapp/static/js/src/services/auth.js

function persistToken(token) {
  if (!token) return;
  try {
    window.localStorage.setItem("mongars_jwt", token);
  } catch (err) {
    console.warn("Unable to persist JWT in localStorage", err);
  }
}

async function getJwt() {
  try {
    const stored = window.localStorage.getItem("mongars_jwt");
    if (stored) {
      return stored;
    }
  } catch (err) {
    console.warn("Unable to read JWT from localStorage", err);
  }
  if (config.token) {
    return config.token;
  }
  throw new Error("Missing JWT (store it in localStorage as 'mongars_jwt').");
}

  if (config.token) {
    persistToken(config.token);
  }

  async function getJwt() {
    try {
      const stored = window.localStorage.getItem("jwt");
      if (stored) {
        return stored;
      }
    } catch (err) {
      console.warn("Unable to read JWT from localStorage", err);
    }
    if (config.token) {
      return config.token;
    }
    throw new Error("Missing JWT (store it in localStorage as 'jwt').");
  }

  return {
    getJwt,
    persistToken,
  };
}
