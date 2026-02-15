import type { ReactNode } from 'react';

interface PanelProps {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
  noPad?: boolean;
}

export function Panel({ title, subtitle, children, className = '', actions, noPad }: PanelProps) {
  return (
    <div className={`rounded-lg border border-border bg-surface ${className}`}>
      {(title || actions) && (
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div>
            {title && <h3 className="text-sm font-semibold text-text-primary">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={noPad ? '' : 'p-4'}>{children}</div>
    </div>
  );
}
