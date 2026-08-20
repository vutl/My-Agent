import { useEffect, useMemo, useState } from "react";

import {
  indexFile,
  indexFolder,
  lightragInsertAll,
  listCollections,
  listDocuments,
  listIndexedFolders,
  vectorIndexAll
} from "../lib/api";
import type {
  CatalogCollection,
  IndexedDocument,
  IndexedFolder,
  IndexFolderResult
} from "../lib/types";

const LIBRARY_PATH_STORAGE_KEY = "aya.library_path";

function loadLibraryPath(): string {
  return localStorage.getItem(LIBRARY_PATH_STORAGE_KEY) ?? import.meta.env.VITE_LIBRARY_PATH ?? "";
}

export function FilesPage() {
  const [libraryPath, setLibraryPath] = useState(loadLibraryPath);
  const [recursive, setRecursive] = useState(false);
  const [status, setStatus] = useState("ready");
  const [folders, setFolders] = useState<IndexedFolder[]>([]);
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [collections, setCollections] = useState<CatalogCollection[]>([]);
  const [lastResult, setLastResult] = useState<IndexFolderResult | null>(null);
  const [vectorStatus, setVectorStatus] = useState("");
  const [graphStatus, setGraphStatus] = useState("");
  const [isBusy, setIsBusy] = useState(false);

  const metrics = useMemo(() => {
    return {
      documents: documents.length,
      chunks: documents.reduce((total, document) => total + document.chunk_count, 0),
      tables: documents.reduce((total, document) => total + (document.table_count ?? 0), 0),
      figures: documents.reduce((total, document) => total + (document.figure_count ?? 0), 0)
    };
  }, [documents]);

  const recentDocuments = useMemo(() => {
    return [...documents]
      .sort((left, right) => Date.parse(right.indexed_at) - Date.parse(left.indexed_at))
      .slice(0, 18);
  }, [documents]);

  useEffect(() => {
    refresh();
  }, []);

  async function refresh(nextStatus = "ready") {
    try {
      const [folderList, documentList, collectionList] = await Promise.all([
        listIndexedFolders(),
        listDocuments(),
        listCollections()
      ]);
      setFolders(folderList);
      setDocuments(documentList);
      setCollections(collectionList);
      setStatus(nextStatus);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "failed");
    }
  }

  async function handleIndex() {
    const selectedPath = libraryPath.trim();
    if (!selectedPath) {
      return;
    }

    setIsBusy(true);
    setStatus("parsing source into SQLite/artifacts");
    setVectorStatus("");
    try {
      await indexCurrentPath(selectedPath);
      localStorage.setItem(LIBRARY_PATH_STORAGE_KEY, selectedPath);
      await refresh("source parsed · retrieval indexes unchanged");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "failed");
    } finally {
      setIsBusy(false);
    }
  }

  async function handleBuildGraphAll() {
    setIsBusy(true);
    setStatus("inserting all parsed documents into LightRAG");
    setGraphStatus("running");
    try {
      const latestDocuments = await listDocuments();
      if (latestDocuments.length === 0) {
        setGraphStatus("no parsed documents");
        setStatus("nothing to insert");
        return;
      }
      // Omit limit for a canonical full sweep so the backend can prune stale
      // graph document IDs left behind by reparse/reindex UUID changes.
      const result = await lightragInsertAll();
      const pruneSummary = result.prune_skipped_reason
        ? ` · stale prune blocked (${result.unready_document_ids?.length ?? 0} unready)`
        : ` · ${result.deleted_stale ?? 0} stale removed`;
      setGraphStatus(
        `${result.inserted}/${result.total} graph indexed · ${result.skipped} skipped · ${result.failed} failed${pruneSummary}`
      );
      await refresh("LightRAG insert finished");
    } catch (error) {
      const message = error instanceof Error ? error.message : "failed";
      setGraphStatus(message);
      setStatus(message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleVectorIndexAll() {
    setIsBusy(true);
    setStatus("rebuilding Lance text/table/figure vectors");
    setVectorStatus("running");
    try {
      const latestDocuments = await listDocuments();
      if (latestDocuments.length === 0) {
        setVectorStatus("no parsed documents");
        setStatus("nothing to rebuild");
        return;
      }
      // A limited run is intentionally non-destructive. The Library action is
      // a full rebuild and must therefore omit limit to prune stale vectors.
      const result = await vectorIndexAll();
      setVectorStatus(
        `${result.indexed_documents}/${result.documents} indexed · ${result.skipped_documents ?? 0} skipped · ${result.failed_documents} failed`
      );
      await refresh("Lance vector rebuild finished");
    } catch (error) {
      const message = error instanceof Error ? error.message : "failed";
      setVectorStatus(message);
      setStatus(message);
    } finally {
      setIsBusy(false);
    }
  }

  async function indexCurrentPath(selectedPath: string) {
    if (/\.(txt|md|pdf|docx)$/i.test(selectedPath)) {
      await indexFile({
        source_path: selectedPath,
        collection_name: selectedPath.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") || "selected_file"
      });
      setLastResult(null);
      return;
    }

    const result = await indexFolder({
      folder_path: selectedPath,
      recursive,
      file_types: ["txt", "md", "pdf", "docx"]
    });
    setLastResult(result);
  }

  return (
    <section className="library-page">
      <header className="library-hero">
        <div className="library-hero-copy">
          <h2>Document Library</h2>
          <p>Index PDFs, DOCX, Markdown once — then ask from Chat.</p>
        </div>
        <div className="library-hero-actions">
          <div className="library-metrics" aria-label="Library metrics">
            <MetricChip label="Docs" value={metrics.documents} />
            <MetricChip label="Chunks" value={metrics.chunks} />
            <MetricChip label="Figures" value={metrics.figures} />
            <MetricChip label="Tables" value={metrics.tables} />
          </div>
          <button className="secondary-button" disabled={isBusy} onClick={() => void refresh()} type="button">
            Refresh
          </button>
        </div>
      </header>

      <section className="library-action-card">
        <label className="library-path-field">
          <span>Source path</span>
          <input
            className="library-path-input"
            value={libraryPath}
            disabled={isBusy}
            onChange={(event) => {
              const nextPath = event.target.value;
              setLibraryPath(nextPath);
              if (nextPath.trim()) localStorage.setItem(LIBRARY_PATH_STORAGE_KEY, nextPath);
              else localStorage.removeItem(LIBRARY_PATH_STORAGE_KEY);
            }}
            placeholder="Choose a folder or a single PDF/DOCX/Markdown file"
          />
        </label>

        <p className="library-index-note">
          <strong>1 · Parse source</strong> writes canonical documents, chunks, tables, figures and artifacts.
          <strong>2 · LightRAG</strong> and <strong>3 · Lance</strong> are separate derived indexes; run them
          explicitly after source changes.
        </p>

        <div className="library-action-row">
          <label className="library-checkbox">
            <input
              checked={recursive}
              disabled={isBusy}
              onChange={(event) => setRecursive(event.target.checked)}
              type="checkbox"
            />
            <span>Recursive folder scan</span>
          </label>

          <div className="library-action-buttons">
            <button
              className="primary-button"
              disabled={isBusy || !libraryPath.trim()}
              onClick={handleIndex}
              type="button"
            >
              1 · Parse / update source
            </button>
            <button className="secondary-button" disabled={isBusy || documents.length === 0} onClick={handleBuildGraphAll} type="button">
              2 · Insert all into LightRAG
            </button>
            <button className="secondary-button" disabled={isBusy || documents.length === 0} onClick={handleVectorIndexAll} type="button">
              3 · Rebuild all Lance vectors
            </button>
          </div>
        </div>

        <div className="library-status-row">
          <span className={`library-status-pill${isBusy ? " busy" : ""}`}>Status: {status}</span>
          {lastResult ? (
            <span className="library-status-pill">
              scanned {lastResult.scanned_files} · indexed {lastResult.indexed_files} · skipped{" "}
              {lastResult.skipped_files}
            </span>
          ) : null}
          {vectorStatus ? <span className="library-status-pill">vectors: {vectorStatus}</span> : null}
          {graphStatus ? <span className="library-status-pill">graph: {graphStatus}</span> : null}
        </div>
      </section>

      <div className="library-content-grid">
        <section className="library-panel document-panel">
          <div className="panel-title-row">
            <h3>Indexed documents</h3>
            <span>{documents.length}</span>
          </div>
          {recentDocuments.length === 0 ? (
            <div className="trace-empty">No indexed documents</div>
          ) : (
            <div className="document-list">
              {recentDocuments.map((document) => (
                <article className="document-row" key={document.id}>
                  <div className="document-row-main">
                    <strong title={document.filename}>{document.filename}</strong>
                    <div className="document-meta">
                      <span>{document.parser_name ?? document.file_type}</span>
                      {document.page_count ? <span>{document.page_count} pages</span> : null}
                      <span>{document.chunk_count} chunks</span>
                      {document.figure_count ? <span>{document.figure_count} figures</span> : null}
                      {document.table_count ? <span>{document.table_count} tables</span> : null}
                    </div>
                  </div>
                  <span className="status-pill">{document.index_status ?? "indexed"}</span>
                </article>
              ))}
            </div>
          )}
        </section>

        <aside className="library-side-stack">
          <section className="library-panel">
            <div className="panel-title-row">
              <h3>Collections</h3>
              <span>{collections.length}</span>
            </div>
            <div className="simple-list">
              {collections.length === 0 ? (
                <div className="trace-empty">No collections</div>
              ) : (
                collections.slice(0, 12).map((collection) => (
                  <div className="simple-list-row" key={collection.id}>
                    <strong>{collection.name}</strong>
                    <span>{collection.type}</span>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="library-panel">
            <div className="panel-title-row">
              <h3>Indexed paths</h3>
              <span>{folders.length}</span>
            </div>
            <div className="simple-list">
              {folders.length === 0 ? (
                <div className="trace-empty">No paths</div>
              ) : (
                folders.slice(0, 8).map((folder) => (
                  <div className="simple-list-row" key={folder.id}>
                    <strong title={folder.folder_path}>{folder.folder_path}</strong>
                    <span>{folder.recursive ? "recursive" : "top level"}</span>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </div>
    </section>
  );
}

function MetricChip({ label, value }: { label: string; value: number }) {
  return (
    <div className="library-metric-chip">
      <span>{label}</span>
      <strong>{value.toLocaleString()}</strong>
    </div>
  );
}
