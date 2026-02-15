export function LoadingSpinner({ text = 'Loading...' }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12">
      <div className="relative h-10 w-10">
        <div className="absolute inset-0 rounded-full border-2 border-border opacity-30" />
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-accent" />
      </div>
      <span className="text-sm text-text-secondary">{text}</span>
    </div>
  );
}
