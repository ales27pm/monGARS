export function createAuthService(config) {
  function persistToken(token) {
    if (!token) return;
    try {
      window.localStorage.setItem("jwt", token);
    } catch (err) {
      console.warn("Unable to persist JWT in localStorage", err);
    }
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
