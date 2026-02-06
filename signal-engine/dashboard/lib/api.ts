const DEFAULT_DOCKER_API_BASE = "http://api:8000";
const DEFAULT_DEV_API_BASE = "http://127.0.0.1:8000";

const resolveApiBase = (): string => {
  const envBase =
    process.env.NEXT_PUBLIC_API_BASE ??
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    "";
  if (envBase) {
    return envBase;
  }

  if (typeof window === "undefined") {
    return process.env.INTERNAL_API_BASE_URL ?? DEFAULT_DOCKER_API_BASE;
  }

  const isLocalhost =
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  return isLocalhost ? DEFAULT_DEV_API_BASE : DEFAULT_DOCKER_API_BASE;
};

export const API_BASE = resolveApiBase();

export type ApiError = {
  status: number;
  message: string;
  url: string;
};

export type EngineStatus = {
  running: boolean;
  mode: string;
  symbols: string[];
  last_heartbeat_ts: string | null;
  last_action: {
    type: string;
    ts: string | null;
    detail?: string;
  } | null;
};

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers ?? {}),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to fetch";
    const apiError: ApiError = { status: 0, message, url };
    throw apiError;
  }

  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload?.detail ?? payload?.message ?? message;
    } catch (error) {
      message = response.statusText;
    }
    const apiError: ApiError = { status: response.status, message, url };
    throw apiError;
  }

  return (await response.json()) as T;
}

export async function getEngineStatus(): Promise<EngineStatus> {
  return apiFetch<EngineStatus>("/engine/status");
}
