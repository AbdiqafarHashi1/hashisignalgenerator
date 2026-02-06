export const API_BASE =
  typeof window === "undefined"
    ? process.env.INTERNAL_API_BASE_URL ?? "http://api:8000"
    : process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type ApiError = {
  status: number;
  message: string;
};

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = payload?.detail ?? payload?.message ?? message;
    } catch (error) {
      message = response.statusText;
    }
    const apiError: ApiError = { status: response.status, message };
    throw apiError;
  }

  return (await response.json()) as T;
}
