import type { ReactElement } from "react";
import type { AppRoute } from "../../App";

interface SidebarProps {
  activeRoute: AppRoute;
  onRouteChange: (route: AppRoute) => void;
}

function IconChat({ active }: { active: boolean }) {
  return (
    <svg className="nav-icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M10 2C5.58 2 2 5.13 2 9c0 1.9.82 3.62 2.16 4.88L3.5 17l3.6-1.44A8.4 8.4 0 0 0 10 16c4.42 0 8-3.13 8-7s-3.58-7-8-7Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        fill={active ? "currentColor" : "none"}
        fillOpacity={active ? 0.15 : 0}
      />
    </svg>
  );
}

function IconLibrary({ active }: { active: boolean }) {
  return (
    <svg className="nav-icon" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="3" y="3" width="5" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.5" fill={active ? "currentColor" : "none"} fillOpacity={active ? 0.15 : 0} />
      <rect x="10" y="3" width="3" height="14" rx="1.5" stroke="currentColor" strokeWidth="1.5" fill={active ? "currentColor" : "none"} fillOpacity={active ? 0.15 : 0} />
      <rect x="15" y="3" width="2" height="14" rx="1" stroke="currentColor" strokeWidth="1.5" fill={active ? "currentColor" : "none"} fillOpacity={active ? 0.15 : 0} />
    </svg>
  );
}

const items: Array<{ route: AppRoute; label: string; icon: (active: boolean) => ReactElement }> = [
  { route: "chat",    label: "Chat",    icon: (a) => <IconChat active={a} /> },
  { route: "library", label: "Library", icon: (a) => <IconLibrary active={a} /> },
];

export function Sidebar({ activeRoute, onRouteChange }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">A</div>
      </div>

      <nav className="nav-list" aria-label="Primary">
        {items.map((item) => {
          const active = activeRoute === item.route;
          return (
            <button
              key={item.route}
              className={active ? "nav-item active" : "nav-item"}
              onClick={() => onRouteChange(item.route)}
              type="button"
              title={item.label}
              aria-label={item.label}
            >
              {item.icon(active)}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
