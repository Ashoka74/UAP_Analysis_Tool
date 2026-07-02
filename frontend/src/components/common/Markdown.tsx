import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

// Theme-styled markdown renderer for AI interpretations and reports (headers,
// lists, bold, blockquotes, fenced code, GFM tables) on the app's dark theme.
const components: Components = {
  h1: ({ node, ...p }) => <h1 className="mb-2 mt-3 text-sm font-bold text-text-primary first:mt-0" {...p} />,
  h2: ({ node, ...p }) => <h2 className="mb-1.5 mt-3 text-[13px] font-semibold text-text-primary first:mt-0" {...p} />,
  h3: ({ node, ...p }) => <h3 className="mb-1 mt-2.5 text-xs font-semibold text-text-primary first:mt-0" {...p} />,
  p: ({ node, ...p }) => <p className="mb-2 last:mb-0" {...p} />,
  ul: ({ node, ...p }) => <ul className="mb-2 ml-4 list-disc space-y-0.5" {...p} />,
  ol: ({ node, ...p }) => <ol className="mb-2 ml-4 list-decimal space-y-0.5" {...p} />,
  li: ({ node, ...p }) => <li className="marker:text-text-muted" {...p} />,
  strong: ({ node, ...p }) => <strong className="font-semibold text-text-primary" {...p} />,
  em: ({ node, ...p }) => <em className="italic" {...p} />,
  a: ({ node, ...p }) => <a className="text-accent hover:underline" target="_blank" rel="noreferrer" {...p} />,
  blockquote: ({ node, ...p }) => <blockquote className="my-2 border-l-2 border-border pl-3 text-text-muted" {...p} />,
  hr: () => <hr className="my-3 border-border" />,
  pre: ({ node, ...p }) => (
    <pre className="my-2 overflow-auto rounded bg-deep p-2.5 text-[11px] leading-relaxed text-text-secondary" {...p} />
  ),
  code: ({ node, className, children, ...p }) => {
    const isBlock = /language-/.test(className || '');
    return isBlock ? (
      <code className={`font-mono ${className ?? ''}`} {...p}>{children}</code>
    ) : (
      <code className="rounded bg-deep px-1 py-0.5 font-mono text-[11px] text-accent-bright" {...p}>{children}</code>
    );
  },
  table: ({ node, ...p }) => (
    <div className="my-2 overflow-auto">
      <table className="w-full border-collapse text-[11px]" {...p} />
    </div>
  ),
  thead: ({ node, ...p }) => <thead className="bg-raised" {...p} />,
  th: ({ node, ...p }) => <th className="border border-border px-2 py-1 text-left font-semibold text-text-secondary" {...p} />,
  td: ({ node, ...p }) => <td className="border border-border px-2 py-1 text-text-secondary" {...p} />,
};

export function Markdown({ children, className = '' }: { children: string; className?: string }) {
  return (
    <div className={`text-xs leading-relaxed text-text-secondary ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
