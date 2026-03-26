import axios from 'axios';
import type { QueryResponse, IngestResponse, DeleteResponse, RebuildResponse, TokenResponse, SuggestQueriesResponse, DocumentListResponse } from './types';

const client = axios.create({ baseURL: '/api' });

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const api = {
  login: async (username: string, password: string): Promise<TokenResponse> => {
    const res = await client.post<TokenResponse>('/auth/token', { username, password });
    return res.data;
  },

  query: async (query: string, topK = 5): Promise<QueryResponse> => {
    const res = await client.post<QueryResponse>('/query', { query, top_k: topK });
    return res.data;
  },

  ingest: async (file: File): Promise<IngestResponse> => {
    const form = new FormData();
    form.append('file', file);
    const res = await client.post<IngestResponse>('/ingest', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  deleteDocument: async (documentId: string): Promise<DeleteResponse> => {
    const res = await client.delete<DeleteResponse>(`/document/${encodeURIComponent(documentId)}`);
    return res.data;
  },

  rebuildIndex: async (): Promise<RebuildResponse> => {
    const res = await client.post<RebuildResponse>('/admin/rebuild-index');
    return res.data;
  },

  suggestQueries: async (): Promise<SuggestQueriesResponse> => {
    const res = await client.get<SuggestQueriesResponse>('/suggest-queries');
    return res.data;
  },

  getDocuments: async (): Promise<DocumentListResponse> => {
    const res = await client.get<DocumentListResponse>('/documents');
    return res.data;
  },
};
