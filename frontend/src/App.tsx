import { lazy, Suspense, useEffect, useState } from 'react';
import { NavLink, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom';
import {
  BookOpen, ChartNoAxesCombined, ChevronLeft, ChevronRight, CircleHelp,
  Clock3, LayoutDashboard, Menu, Moon, Settings, ShieldCheck, Sparkles, Sun, Users, X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { api } from './api';

const OverviewPage    = lazy(() => import('./pages/OverviewPage').then(m => ({ default: m.OverviewPage })));
const PredictionPage  = lazy(() => import('./pages/PredictionPage').then(m => ({ default: m.PredictionPage })));
const PerformancePage = lazy(() => import('./pages/PerformancePage').then(m => ({ default: m.PerformancePage })));
const InsightsPage    = lazy(() => import('./pages/InsightsPage').then(m => ({ default: m.InsightsPage })));
const HistoryPage     = lazy(() => import('./pages/HistoryPage').then(m => ({ default: m.HistoryPage })));
const AboutPage       = lazy(() => import('./pages/InfoPages').then(m => ({ default: m.AboutPage })));
const SettingsPage    = lazy(() => import('./pages/InfoPages').then(m => ({ default: m.SettingsPage })));

type Theme = 'light' | 'dark';
type NavItem = { label: string; path: string; icon: LucideIcon };

const nav: NavItem[] = [
  { label: 'Overview',           path: '/',           icon: LayoutDashboard },
  { label: 'Predict Attrition',  path: '/predict',    icon: Sparkles },
  { label: 'Model Performance',  path: '/performance', icon: ChartNoAxesCombined },
  { label: 'Employee Insights',  path: '/insights',   icon: Users },
  { label: 'Prediction History', path: '/history',    icon: Clock3 },
];

function ShimmerFallback() {
  return (
    <div className="page-stack">
      <div style={{ height: 52 }} className="shimmer-block shimmer-line" />
      <div className="grid-4">
        {[0,1,2,3].map(i => (
          <div key={i} className="shimmer-card">
            <div className="shimmer-line shimmer-block h-sm w-60" />
            <div className="shimmer-line shimmer-block h-lg" />
            <div className="shimmer-line shimmer-block h-sm w-40" />
          </div>
        ))}
      </div>
      <div className="grid-2">
        {[0,1].map(i => (
          <div key={i} className="shimmer-card" style={{ height: 280 }}>
            <div style={{ width: 200, height: 10 }} className="shimmer-line shimmer-block" />
          </div>
        ))}
      </div>
    </div>
  );
}

function AppRoutes() {
  const [theme, setTheme] = useState<Theme>(() =>
    localStorage.getItem('attritioniq-theme') === 'light' ? 'light' : 'dark'
  );
  const [service, setService] = useState<{ status: string; available_models: number } | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('attritioniq-theme', theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    const check = () =>
      api.health()
        .then(h => { if (active) setService({ status: h.status, available_models: h.available_models }); })
        .catch(() => { if (active) setService({ status: 'unavailable', available_models: 0 }); });
    void check();
    const t = setInterval(check, 45_000);
    return () => { active = false; clearInterval(t); };
  }, []);

  return (
    <Routes>
      <Route element={<Shell theme={theme} setTheme={setTheme} service={service} />}>
        <Route index           element={<OverviewPage />} />
        <Route path="predict"     element={<PredictionPage />} />
        <Route path="performance" element={<PerformancePage />} />
        <Route path="insights"    element={<InsightsPage />} />
        <Route path="history"     element={<HistoryPage />} />
        <Route path="about"       element={<AboutPage />} />
        <Route path="settings"    element={<SettingsPage theme={theme} setTheme={setTheme} service={service} />} />
        <Route path="*"           element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function Shell({
  theme, setTheme, service,
}: {
  theme: Theme;
  setTheme: (t: Theme) => void;
  service: { status: string; available_models: number } | null;
}) {
  const loc = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => setMobileOpen(false), [loc.pathname]);

  const current = nav.find(n => n.path === loc.pathname);
  const title   = current?.label ?? (loc.pathname === '/about' ? 'About the Model' : 'Settings');
  const online  = service?.status === 'ok';

  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {mobileOpen && (
        <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />
      )}

      {/* ── SIDEBAR ── */}
      <aside className={`sidebar ${mobileOpen ? 'open' : ''}`} aria-label="Primary navigation">
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden="true">A</div>
          {!collapsed && (
            <div className="brand-copy">
              <strong>AttritionIQ</strong>
              <span>People Analytics</span>
            </div>
          )}
          <button
            className="icon-btn sidebar-close"
            aria-label="Close navigation"
            onClick={() => setMobileOpen(false)}
            style={{ marginLeft: 'auto' }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Main nav */}
        <div className="sidebar-section">
          {!collapsed && <div className="sidebar-section-label">Workspace</div>}
          <nav className="nav-list">
            {nav.map(({ label, path, icon: Icon }) => (
              <NavLink
                key={path}
                to={path}
                end={path === '/'}
                title={collapsed ? label : undefined}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
              >
                <Icon size={16} strokeWidth={1.7} aria-hidden="true" />
                {!collapsed && <span>{label}</span>}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Reference nav */}
        <div className="sidebar-section">
          {!collapsed && <div className="sidebar-section-label">Reference</div>}
          <nav className="nav-list">
            <NavLink
              to="/about"
              title={collapsed ? 'About' : undefined}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              <BookOpen size={16} strokeWidth={1.7} />
              {!collapsed && <span>About the Model</span>}
            </NavLink>
            <NavLink
              to="/settings"
              title={collapsed ? 'Settings' : undefined}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              <Settings size={16} strokeWidth={1.7} />
              {!collapsed && <span>Settings</span>}
            </NavLink>
          </nav>
        </div>

        {/* Bottom */}
        <div className="sidebar-bottom">
          <div className={`service-mini ${online ? 'is-online' : 'is-offline'}`}>
            <span className="status-dot" />
            {!collapsed && (
              <span>
                {online
                  ? `${service?.available_models ?? 0} models ready`
                  : 'Service offline'}
              </span>
            )}
          </div>
          <button
            className="collapse-button"
            onClick={() => setCollapsed(c => !c)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed
              ? <ChevronRight size={15} />
              : <><ChevronLeft size={15} /><span>Collapse</span></>}
          </button>
        </div>
      </aside>

      {/* ── MAIN ── */}
      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-btn mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
            >
              <Menu size={18} />
            </button>
            <div className="breadcrumb">
              <span>AttritionIQ</span>
              <ChevronRight size={11} className="breadcrumb-sep" />
              <strong>{title}</strong>
            </div>
          </div>
          <div className="topbar-right">
            <div className={`service-pill ${online ? 'is-online' : 'is-offline'}`}>
              <span className="status-dot" />
              <span>{online ? 'Service live' : 'Offline'}</span>
            </div>
            <button
              className="icon-btn"
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
              onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            >
              {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
            </button>
            <div className="profile-chip" aria-label="Workspace">
              <div className="profile-avatar">AI</div>
              <span>Workspace</span>
            </div>
          </div>
        </header>

        <main className="page-content">
          <Suspense fallback={<ShimmerFallback />}>
            <Outlet />
          </Suspense>
        </main>

        <footer className="app-footer">
          <ShieldCheck size={13} />
          <span>Decision support only — not for employment decisions</span>
          <a href="/docs" target="_blank" rel="noreferrer">
            <CircleHelp size={13} /> API docs
          </a>
        </footer>
      </div>
    </div>
  );
}

export default AppRoutes;