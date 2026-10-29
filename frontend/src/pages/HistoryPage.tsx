import { useState } from 'react';
import { ChevronLeft, ChevronRight, Search, Trash2, X } from 'lucide-react';
import type { HistoryEntry } from '../types';
import { EmptyState, modelLabel, PageHeader, percent } from '../components/ui';

const PAGE_SIZE = 8;

function readHistory(): HistoryEntry[] {
  try { return JSON.parse(sessionStorage.getItem('attritioniq-history') ?? '[]') as HistoryEntry[]; }
  catch { return []; }
}

export function HistoryPage() {
  const [history, setHistory] = useState<HistoryEntry[]>(readHistory);
  const [query,   setQuery]   = useState('');
  const [outcome, setOutcome] = useState('all');
  const [sort,    setSort]    = useState<'newest' | 'oldest' | 'probability'>('newest');
  const [page,    setPage]    = useState(0);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);

  const filtered = history
    .filter(item => {
      const q = `${modelLabel(item.model)} ${item.id}`.toLowerCase().includes(query.toLowerCase());
      const o = outcome === 'all' || item.prediction === outcome;
      return q && o;
    })
    .sort((a, b) =>
      sort === 'oldest'      ? a.createdAt.localeCompare(b.createdAt)
      : sort === 'probability' ? b.probability - a.probability
      : b.createdAt.localeCompare(a.createdAt)
    );

  const pages    = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const fmt      = (d: string) => new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(d));

  function clearHistory() {
    sessionStorage.removeItem('attritioniq-history');
    setHistory([]); setPage(0);
  }

  return (
    <div className="page-stack fade-in">
      <PageHeader
        eyebrow="Workspace"
        title="Prediction History"
        description="Recent prediction summaries from this browser session. Employee inputs are not stored."
        action={
          history.length > 0
            ? <button className="btn btn-secondary" onClick={clearHistory}><Trash2 size={14} /> Clear</button>
            : undefined
        }
      />

      <section className="panel" style={{ padding: 0, overflow: 'hidden' }}>
        <div className="history-toolbar">
          <label className="search-wrap" aria-label="Search">
            <Search size={14} />
            <input
              aria-label="Search prediction history"
              value={query}
              onChange={e => { setQuery(e.target.value); setPage(0); }}
              placeholder="Search by model or ID…"
            />
          </label>
          <div className="filter-row">
            <label className="visually-hidden" htmlFor="hist-outcome">Outcome</label>
            <select
              id="hist-outcome" value={outcome}
              onChange={e => { setOutcome(e.target.value); setPage(0); }}
            >
              <option value="all">All outcomes</option>
              <option value="Yes">At risk</option>
              <option value="No">Likely to stay</option>
            </select>
            <label className="visually-hidden" htmlFor="hist-sort">Sort</label>
            <select id="hist-sort" value={sort} onChange={e => setSort(e.target.value as typeof sort)}>
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="probability">Highest probability</option>
            </select>
          </div>
        </div>

        {pageRows.length === 0 ? (
          <EmptyState
            title={history.length ? 'No matching predictions' : 'No predictions this session'}
            description={history.length
              ? 'Try changing your search or filter.'
              : 'Completed predictions appear here for this browser session.'}
          />
        ) : (
          <>
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th><th>Date & time</th><th>Model</th>
                    <th>Outcome</th><th>Probability</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map(item => (
                    <tr key={item.id}>
                      <td className="id-cell">{item.id.slice(0, 8)}</td>
                      <td>{fmt(item.createdAt)}</td>
                      <td className="model-cell">{modelLabel(item.model)}</td>
                      <td>
                        <span className={`table-outcome ${item.prediction === 'Yes' ? 'risk' : 'stay'}`}>
                          {item.prediction === 'Yes' ? 'At risk' : 'Likely to stay'}
                        </span>
                      </td>
                      <td className="tabular">{percent(item.probability, 1)}</td>
                      <td>
                        <button className="text-link" onClick={() => setSelected(item)}>View</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="table-footer">
              <span>{filtered.length} prediction{filtered.length === 1 ? '' : 's'}</span>
              <div className="pagination">
                <button
                  className="icon-btn" aria-label="Previous page"
                  onClick={() => setPage(p => Math.max(0, p-1))}
                  disabled={page === 0}
                >
                  <ChevronLeft size={15} />
                </button>
                <span>Page {page+1} of {pages}</span>
                <button
                  className="icon-btn" aria-label="Next page"
                  onClick={() => setPage(p => Math.min(pages-1, p+1))}
                  disabled={page >= pages-1}
                >
                  <ChevronRight size={15} />
                </button>
              </div>
            </div>
          </>
        )}
      </section>

      <p className="source-note">
        History contains only model, timestamp, outcome, and probability — no employee profile fields or API keys.
      </p>

      {/* Detail dialog */}
      {selected && (
        <div
          className="dialog-backdrop"
          onMouseDown={e => { if (e.target === e.currentTarget) setSelected(null); }}
        >
          <section
            className="dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="hist-detail-title"
          >
            <div className="dialog-head">
              <div>
                <div className="page-eyebrow">Prediction record</div>
                <h2 id="hist-detail-title">{selected.id.slice(0,8)}</h2>
              </div>
              <button className="icon-btn" aria-label="Close" onClick={() => setSelected(null)}>
                <X size={16} />
              </button>
            </div>

            <dl className="def-list">
              <div><dt>Date & time</dt><dd>{fmt(selected.createdAt)}</dd></div>
              <div><dt>Model</dt><dd>{modelLabel(selected.model)}</dd></div>
              <div>
                <dt>Outcome</dt>
                <dd>
                  <span className={`tag ${selected.prediction === 'Yes' ? 'red' : 'green'}`}>
                    {selected.prediction === 'Yes' ? 'At risk' : 'Likely to stay'}
                  </span>
                </dd>
              </div>
              <div>
                <dt>Attrition probability</dt>
                <dd style={{ color: selected.prediction === 'Yes' ? 'var(--red)' : 'var(--green)' }}>
                  {percent(selected.probability, 1)}
                </dd>
              </div>
            </dl>
            <p className="panel-footnote">Employee profile fields are not retained in session history.</p>
          </section>
        </div>
      )}
    </div>
  );
}