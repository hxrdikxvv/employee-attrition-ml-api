import { lazy, Suspense, useEffect, useState } from 'react';
import { NavLink, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom';
import {
  BookOpen, ChartNoAxesCombined, ChevronLeft, ChevronRight, CircleHelp, LoaderCircle,
  Clock3, LayoutDashboard, Menu, Moon, Settings, ShieldCheck, Sparkles, Sun, Users, X,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { api } from './api';

const OverviewPage = lazy(() => import('./pages/OverviewPage').then((module) => ({ default: module.OverviewPage })));
const PredictionPage = lazy(() => import('./pages/PredictionPage').then((module) => ({ default: module.PredictionPage })));
const PerformancePage = lazy(() => import('./pages/PerformancePage').then((module) => ({ default: module.PerformancePage })));
const InsightsPage = lazy(() => import('./pages/InsightsPage').then((module) => ({ default: module.InsightsPage })));
const HistoryPage = lazy(() => import('./pages/HistoryPage').then((module) => ({ default: module.HistoryPage })));
const AboutPage = lazy(() => import('./pages/InfoPages').then((module) => ({ default: module.AboutPage })));
const SettingsPage = lazy(() => import('./pages/InfoPages').then((module) => ({ default: module.SettingsPage })));

type Theme = 'light' | 'dark';
type NavItem = { label: string; path: string; icon: LucideIcon };

const navigation: NavItem[] = [
  { label: 'Overview', path: '/', icon: LayoutDashboard },
  { label: 'Predict Attrition', path: '/predict', icon: Sparkles },
  { label: 'Model Performance', path: '/performance', icon: ChartNoAxesCombined },
  { label: 'Employee Insights', path: '/insights', icon: Users },
  { label: 'Prediction History', path: '/history', icon: Clock3 },
];

function AppRoutes() {
  const [theme, setTheme] = useState<Theme>(() => localStorage.getItem('attritioniq-theme') === 'dark' ? 'dark' : 'light');
  const [service, setService] = useState<{ status: string; available_models: number } | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('attritioniq-theme', theme);
  }, [theme]);

  useEffect(() => {
    let active = true;
    const check = () => api.health().then((health) => {
      if (active) setService({ status: health.status, available_models: health.available_models });
    }).catch(() => {
      if (active) setService({ status: 'unavailable', available_models: 0 });
    });
    void check();
    const timer = window.setInterval(check, 45_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  return (
    <Routes>
      <Route element={<ApplicationShell theme={theme} setTheme={setTheme} service={service} />}>
        <Route index element={<OverviewPage />} />
        <Route path="predict" element={<PredictionPage />} />
        <Route path="performance" element={<PerformancePage />} />
        <Route path="insights" element={<InsightsPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="settings" element={<SettingsPage theme={theme} setTheme={setTheme} service={service} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

function ApplicationShell({
  theme,
  setTheme,
  service,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  service: { status: string; available_models: number } | null;
}) {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const current = navigation.find((item) => item.path === location.pathname);
  const title = current?.label ?? (location.pathname === '/about' ? 'About the Model' : 'Settings');
  const serviceReady = service?.status === 'ok';

  useEffect(() => setMobileOpen(false), [location.pathname]);

  return (
    <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
      <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`} aria-label="Primary navigation">
        <div className="sidebar-brand">
          <div className="brand-mark" aria-hidden="true">A</div>
          {!collapsed && <div className="brand-copy"><strong>AttritionIQ</strong><span>PEOPLE ANALYTICS</span></div>}
          <button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={17} /></button>
        </div>
        <div className="sidebar-group-label">WORKSPACE</div>
        <nav className="nav-list">
          {navigation.map(({ label, path, icon: Icon }) => (
            <NavLink key={path} to={path} end={path === '/'} title={collapsed ? label : undefined} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Icon size={17} strokeWidth={1.8} aria-hidden="true" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-group-label secondary-label">REFERENCE</div>
        <nav className="nav-list">
          <NavLink to="/about" title={collapsed ? 'About the Model' : undefined} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <BookOpen size={17} strokeWidth={1.8} aria-hidden="true" />{!collapsed && <span>About the Model</span>}
          </NavLink>
          <NavLink to="/settings" title={collapsed ? 'Settings' : undefined} onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <Settings size={17} strokeWidth={1.8} aria-hidden="true" />{!collapsed && <span>Settings</span>}
          </NavLink>
        </nav>
        <div className="sidebar-bottom">
          <div className={`service-mini ${serviceReady ? 'is-online' : 'is-offline'}`}>
            <span className="status-dot" />
            {!collapsed && <span>{serviceReady ? `${service?.available_models ?? 0} models available` : 'Service unavailable'}</span>}
          </div>
          <button className="collapse-button" onClick={() => setCollapsed(!collapsed)} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {collapsed ? <ChevronRight size={16} /> : <><ChevronLeft size={16} /><span>Collapse sidebar</span></>}
          </button>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button>
            <div className="breadcrumb"><span>AttritionIQ</span><ChevronRight size={13} /><strong>{title}</strong></div>
          </div>
          <div className="topbar-actions">
            <div className={`service-pill ${serviceReady ? 'online' : 'offline'}`}><span className="status-dot" /><span>{serviceReady ? 'Model service live' : 'Service unavailable'}</span></div>
            <button className="icon-button theme-button" aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`} onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
              {theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}
            </button>
            <div className="profile-avatar" aria-label="AttritionIQ workspace">AI</div>
          </div>
        </header>
              <main className="page-content"><Suspense fallback={<div className="state-message"><LoaderCircle className="spin" size={17} />Loading workspace</div>}><Outlet /></Suspense></main>
        <footer className="app-footer"><ShieldCheck size={14} /><span>Decision support only · Use alongside human review</span><a href="/docs"><CircleHelp size={14} /> API docs</a></footer>
      </div>
    </div>
  );
}

export default AppRoutes;