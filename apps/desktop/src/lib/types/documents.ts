export interface IndexedFolder {
  id: string;
  folder_path: string;
  recursive: boolean;
  file_types: string[];
  created_at: string;
  updated_at: string;
}

export interface IndexedDocument {
  id: string;
  folder_id: string;
  source_path: string;
  filename: string;
  file_type: string;
  modified_at: string;
  indexed_at: string;
  chunk_count: number;
  table_count?: number;
  figure_count?: number;
  index_status?: string;
  parser_name?: string | null;
  page_count?: number | null;
}

export interface CatalogCollection {
  id: string;
  name: string;
  type: string;
  scope_type: string;
  scope_id: string | null;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface RetrievedDocument {
  chunk_id: string;
  document_id: string;
  source_path: string;
  filename: string;
  content: string;
  citation_label?: string;
  score: number;
  chunk_index?: number;
  page_number?: number | null;
  heading_path?: string[];
  token_count?: number | null;
  char_count?: number | null;
  retrieval_channels?: string[];
  vector_rank?: number | null;
  fts_rank?: number | null;
  source_id?: string;
  chunk_type?: string | null;
  artifact_type?: string | null;
  caption?: string | null;
  image_path?: string | null;
  image_url?: string | null;
  table_id?: string | null;
  table_index?: number | null;
  figure_id?: string | null;
  figure_index?: number | null;
  figure_label?: string | null;
  figure_number?: number | null;
  quality_status?: string | null;
  asset_kind?: string | null;
  is_content?: boolean | null;
  is_complete?: boolean | null;
  figure_type?: string | null;
}

export interface IndexFolderResult {
  folder_id: string;
  folder_path: string;
  scanned_files: number;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  documents: Array<{
    id: string;
    source_path: string;
    filename: string;
    file_type: string;
    chunk_count: number;
  }>;
}

export interface CatalogFile {
  id: string;
  source_path: string;
  filename: string;
  extension?: string | null;
  size_bytes?: number | null;
  modified_at?: string | null;
  approved: boolean;
  is_supported: boolean;
}

export interface ScanFolderResult {
  folder_id: string;
  folder_path: string;
  recursive: boolean;
  file_count: number;
  supported_files: number;
  unsupported_files: number;
  files: CatalogFile[];
}

export interface CatalogSearchResult {
  source: "file" | "document_card";
  file_id?: string;
  document_id?: string;
  source_path: string;
  filename: string;
  extension?: string | null;
  title_guess?: string | null;
  short_summary?: string | null;
  rank_score?: number;
  is_supported?: boolean;
}

export interface CatalogSearchResponse {
  query: string;
  fts_query: string;
  retrieval_channels?: string[];
  results: CatalogSearchResult[];
}

export interface FileCandidate {
  file_id: string | null;
  filename: string;
  source_path: string;
  extension?: string | null;
  size_bytes?: number | null;
  modified_at?: string | null;
  confidence: number;
  reason: string;
  is_supported: boolean;
}

export interface ResolveFileResponse {
  status: "single_match" | "multiple_matches" | "not_found";
  candidates: FileCandidate[];
}

export interface DirectReadResponse {
  source_path: string;
  filename: string;
  file_type: string;
  parser_name: string;
  parser_version: string;
  page_count: number | null;
  token_estimate: number;
  truncated: boolean;
  content: string;
}

export interface IndexSelectedFilesResult {
  collection_id: string;
  collection_name: string;
  indexed_files: number;
  skipped_files: number;
  failed_files: number;
  documents: IndexFolderResult["documents"];
  errors: Array<{ source_path: string; error: string }>;
}

export interface VectorIndexAllResult {
  documents: number;
  indexed_documents: number;
  skipped_documents?: number;
  failed_documents: number;
}

export interface LightRAGInsertAllResult {
  total: number;
  inserted: number;
  skipped: number;
  failed: number;
  deleted_stale?: number;
  prune_skipped_reason?: string | null;
  unready_document_ids?: string[];
  results: Array<{
    document_id: string;
    ok: boolean;
    track_id?: string;
    source_path?: string;
    char_count?: number;
    skipped?: boolean;
    reason?: string | null;
    error?: string;
  }>;
}

export interface LightRAGStatus {
  enabled: boolean;
  retrieval_engine: string;
  llm_model: string;
  llm_api_base: string;
  embedding_model: string;
  working_dir: string;
  status_counts: Record<string, number>;
  documents: Array<Record<string, unknown>>;
}

export interface HybridSearchResponse {
  query: string;
  collection_id?: string | null;
  selected_document_ids: string[];
  retrieval_channels: string[];
  results: RetrievedDocument[];
  context?: {
    sources: RetrievedDocument[];
    stats: {
      source_count: number;
      character_count: number;
      document_count: number;
    };
  };
}

