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
  const [query, setQuery] = useState('');
  const [outcome, setOutcome] = useState('all');
  const [sort, setSort] = useState<'newest' | 'oldest' | 'probability'>('newest');
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<HistoryEntry | null>(null);

  const filtered = history.filter((item) => {
    const matchesQuery = `${modelLabel(item.model)} ${item.id}`.toLowerCase().includes(query.toLowerCase());
    const matchesOutcome = outcome === 'all' || item.prediction === outcome;
    return matchesQuery && matchesOutcome;
  }).sort((a, b) => sort === 'oldest' ? a.createdAt.localeCompare(b.createdAt) : sort === 'probability' ? b.probability - a.probability : b.createdAt.localeCompare(a.createdAt));

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function clearHistory() {
    sessionStorage.removeItem('attritioniq-history');
    setHistory([]);
    setPage(0);
  }

  return (
    <div className="page-stack">
      <PageHeader eyebrow="Workspace" title="Prediction history" description="Recent prediction summaries from this browser session. Employee inputs are not stored." action={history.length > 0 ? <button className="button button-secondary" onClick={clearHistory}><Trash2 size={15} /> Clear history</button> : undefined} />
      <section className="panel history-panel">
        <div className="history-toolbar">
          <label className="search-control"><Search size={16} /><input aria-label="Search prediction history" value={query} onChange={(event) => { setQuery(event.target.value); setPage(0); }} placeholder="Search model or prediction ID" /></label>
          <div className="history-filters">
            <label className="visually-hidden" htmlFor="history-outcome">Filter by outcome</label>
            <select id="history-outcome" value={outcome} onChange={(event) => { setOutcome(event.target.value); setPage(0); }}><option value="all">All outcomes</option><option value="Yes">At risk</option><option value="No">Likely to stay</option></select>
            <label className="visually-hidden" htmlFor="history-sort">Sort predictions</label>
            <select id="history-sort" value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="probability">Highest probability</option></select>
          </div>
        </div>
        {pageRows.length === 0 ? <EmptyState title={history.length ? 'No matching predictions' : 'No predictions in this session'} description={history.length ? 'Try changing your search or outcome filter.' : 'Completed predictions appear here for this browser session only.'} /> : <>
          <div className="table-scroll"><table className="data-table history-table"><thead><tr><th>Prediction ID</th><th>Date &amp; time</th><th>Model</th><th>Outcome</th><th>Probability</th><th>Details</th></tr></thead><tbody>
            {pageRows.map((item) => <tr key={item.id}><td className="id-cell">{item.id.slice(0, 8)}</td><td>{new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(item.createdAt))}</td><td className="model-cell">{modelLabel(item.model)}</td><td><span className={`table-outcome ${item.prediction === 'Yes' ? 'risk' : 'stay'}`}>{item.prediction === 'Yes' ? 'At risk' : 'Likely to stay'}</span></td><td className="tabular">{percent(item.probability, 1)}</td><td><button className="text-button" onClick={() => setSelected(item)}>View</button></td></tr>)}
          </tbody></table></div>
          <div className="table-footer"><span>{filtered.length} prediction{filtered.length === 1 ? '' : 's'}</span><div className="pagination"><button className="icon-button" onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0} aria-label="Previous page"><ChevronLeft size={16} /></button><span>Page {page + 1} of {pages}</span><button className="icon-button" onClick={() => setPage(Math.min(pages - 1, page + 1))} disabled={page >= pages - 1} aria-label="Next page"><ChevronRight size={16} /></button></div></div>
        </>}
      </section>
      <p className="source-note">History contains only model, timestamp, outcome and probability. It does not include employee profile fields or API keys.</p>
      {selected && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setSelected(null); }}>
        <section className="detail-dialog" role="dialog" aria-modal="true" aria-labelledby="history-detail-title">
          <div className="dialog-heading"><div><span className="page-eyebrow">Prediction record</span><h2 id="history-detail-title">{selected.id.slice(0, 8)}</h2></div><button className="icon-button" aria-label="Close details" onClick={() => setSelected(null)}><X size={17} /></button></div>
          <dl className="definition-list"><div><dt>Date &amp; time</dt><dd>{new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(selected.createdAt))}</dd></div><div><dt>Model</dt><dd>{modelLabel(selected.model)}</dd></div><div><dt>Outcome</dt><dd>{selected.prediction === 'Yes' ? 'At risk of leaving' : 'Likely to stay'}</dd></div><div><dt>Attrition probability</dt><dd>{percent(selected.probability, 1)}</dd></div></dl>
          <p className="panel-footnote">Employee profile fields are not retained in session history.</p>
        </section>
      </div>}
    </div>
  );
}