import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '@/lib/api';
import type { DocumentItem } from '@/lib/types';
import { Upload, Trash2, Loader2, CheckCircle2, XCircle, RefreshCw, ExternalLink, Database } from 'lucide-react';

interface UploadResult {
  filename: string;
  status: 'success' | 'error';
  message: string;
}

export default function DocumentsPage() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadResult[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [docsLoading, setDocsLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res = await api.getDocuments();
      setDocuments(res.documents);
      setTotalChunks(res.total_chunks);
    } catch {
      setDocuments([]);
    } finally {
      setDocsLoading(false);
    }
  }, []);

  useEffect(() => { void loadDocuments(); }, [loadDocuments]);

  const uploadFiles = async (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) =>
      f.name.endsWith('.md') || f.name.endsWith('.txt') || f.name.endsWith('.pdf')
    );
    if (!arr.length) return;
    setUploading(true);
    setResults([]);
    for (const file of arr) {
      try {
        const res = await api.ingest(file);
        setResults((prev) => [...prev, { filename: file.name, status: 'success', message: `${res.chunks_indexed} chunks indexed` }]);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        setResults((prev) => [...prev, { filename: file.name, status: 'error', message: msg }]);
      }
    }
    setUploading(false);
    void loadDocuments();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void uploadFiles(e.dataTransfer.files);
  };

  const handleDelete = async (doc: DocumentItem) => {
    setDeletingId(doc.document_id);
    try {
      await api.deleteDocument(doc.document_id);
      setDocuments((prev) => prev.filter((d) => d.document_id !== doc.document_id));
    } catch {
      // silent — keep the row
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 className="text-white font-semibold">Documents</h2>
          <p className="text-slate-400 text-xs mt-0.5">
            {docsLoading ? 'Loading…' : `${documents.length} papers · ${totalChunks.toLocaleString()} chunks`}
          </p>
        </div>
        <button
          onClick={() => void loadDocuments()}
          disabled={docsLoading}
          className="flex items-center gap-1.5 text-slate-500 hover:text-slate-300 text-xs transition disabled:opacity-40"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${docsLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="px-6 py-6 space-y-8 max-w-3xl">

        {/* Document List */}
        <section>
          <h3 className="text-sm font-medium text-slate-300 mb-3">Indexed Papers</h3>
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
            {docsLoading ? (
              <div className="flex items-center justify-center py-12 gap-2 text-slate-500 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading documents…
              </div>
            ) : documents.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                <Database className="w-8 h-8 mb-2 opacity-40" />
                <p className="text-sm">No documents indexed yet</p>
                <p className="text-xs mt-1 opacity-60">Upload PDFs below to get started</p>
              </div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-left">
                    <th className="px-4 py-2.5 text-slate-500 font-medium text-xs">Paper</th>
                    <th className="px-4 py-2.5 text-slate-500 font-medium text-xs w-20 text-right">Chunks</th>
                    <th className="px-4 py-2.5 w-12"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {documents.map((doc) => (
                    <tr key={doc.document_id} className="group hover:bg-slate-800/30 transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex flex-col gap-0.5">
                          <span className="text-slate-200 text-xs font-medium truncate max-w-xs">
                            {doc.paper_title || doc.source_filename}
                          </span>
                          <div className="flex items-center gap-2">
                            <span className="text-slate-600 text-xs truncate">{doc.source_filename}</span>
                            {doc.arxiv_id && (
                              <a
                                href={`https://arxiv.org/abs/${doc.arxiv_id}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-cyan-600 hover:text-cyan-400 transition flex items-center gap-0.5"
                                onClick={(e) => e.stopPropagation()}
                              >
                                <ExternalLink className="w-3 h-3" />
                                <span className="text-xs">{doc.arxiv_id}</span>
                              </a>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right text-slate-500 text-xs tabular-nums">
                        {doc.chunk_count}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => void handleDelete(doc)}
                          disabled={deletingId === doc.document_id}
                          className="opacity-0 group-hover:opacity-100 flex items-center justify-center w-7 h-7 rounded-lg text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition disabled:opacity-40"
                          title={`Delete ${doc.source_filename}`}
                        >
                          {deletingId === doc.document_id
                            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            : <Trash2 className="w-3.5 h-3.5" />}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Upload Zone */}
        <section>
          <h3 className="text-sm font-medium text-slate-300 mb-3">Upload Documents</h3>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
              dragging ? 'border-indigo-500 bg-indigo-500/8' : 'border-slate-700/60 hover:border-slate-600 hover:bg-slate-800/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              className="hidden"
              onChange={(e) => e.target.files && void uploadFiles(e.target.files)}
            />
            <Upload className="w-10 h-10 text-slate-500 mx-auto mb-3" />
            <p className="text-slate-300 font-medium">Drop files here or click to browse</p>
            <p className="text-slate-500 text-sm mt-1">Supported: PDF, TXT, Markdown</p>
          </div>

          {uploading && (
            <div className="flex items-center gap-2 mt-4 text-indigo-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" /> Uploading and indexing…
            </div>
          )}

          {results.length > 0 && (
            <div className="mt-4 space-y-2">
              {results.map((r, i) => (
                <div
                  key={i}
                  className={`flex items-center gap-3 rounded-lg px-4 py-2.5 text-sm ${
                    r.status === 'success'
                      ? 'bg-green-500/10 border border-green-500/30 text-green-300'
                      : 'bg-red-500/10 border border-red-500/30 text-red-300'
                  }`}
                >
                  {r.status === 'success' ? <CheckCircle2 className="w-4 h-4 shrink-0" /> : <XCircle className="w-4 h-4 shrink-0" />}
                  <span className="font-medium truncate">{r.filename}</span>
                  <span className="text-xs opacity-70 ml-auto shrink-0">{r.message}</span>
                </div>
              ))}
            </div>
          )}
        </section>

      </div>
    </div>
  );
}
