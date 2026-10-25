import type { EmployeeInput, InsightsData, ModelInfo, PerformanceData, PredictionResult } from './types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = typeof payload === 'object' && payload !== null && 'detail' in payload
      ? String(payload.detail)
      : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload as T;
}

export const api = {
  health: () => request<{ status: string; default_model: string; available_models: number }>('/health'),
  models: () => request<{ default_model: string; threshold: number; models: ModelInfo[] }>('/models'),
  performance: () => request<PerformanceData>('/api/performance'),
  insights: () => request<InsightsData>('/api/insights'),
  predict: (model: string, employee: EmployeeInput, apiKey: string) => request<PredictionResult>(
    `/predict/${encodeURIComponent(model)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(employee),
    },
  ),
};

export const API_ORIGIN = API_BASE_URL || window.location.origin;