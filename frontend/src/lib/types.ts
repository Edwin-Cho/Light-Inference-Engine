export interface Citation {
  source_filename: string;
  paper_title?: string;
  arxiv_id?: string;
  section_header?: string;
  page_number?: number;
  score: number;
}

export interface QueryResponse {
  request_id: string;
  answer: string;
  citations: Citation[];
  status: 'ok' | 'partial' | 'no_context';
  warning?: string;
}

export interface IngestResponse {
  filename: string;
  chunks_indexed: number;
  status: string;
}

export interface DeleteResponse {
  filename: string;
  chunks_removed: number;
  status: string;
}

export interface RebuildResponse {
  request_id: string;
  documents_reindexed: number;
  chunks_total: number;
  status: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface SuggestQueriesResponse {
  questions: string[];
  cached: boolean;
}

export interface DocumentItem {
  document_id: string;
  source_filename: string;
  paper_title?: string;
  arxiv_id?: string;
  chunk_count: number;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
  total_chunks: number;
}
