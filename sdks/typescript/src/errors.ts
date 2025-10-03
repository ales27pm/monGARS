export class ApiError extends Error {
  public readonly status: number;
  public readonly payload: unknown;

  constructor(status: number, message: string, payload?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export class AuthenticationError extends ApiError {
  constructor(status: number, message: string, payload?: unknown) {
    super(status, message || "Authentication failed", payload);
    this.name = "AuthenticationError";
  }
}
