"use client";

import { useState } from "react";
import { FileText, ChevronDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Minimal markdown → JSX (headings, tables, lists, blockquote, bold, code).
function renderMarkdown(md: string): React.ReactNode {
  const lines = md.split("\n");
  const out: React.ReactNode[] = [];
  let i = 0;
  const inline = (t: string) =>
    t
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/`(.+?)`/g, '<code class="rounded bg-panel-2 px-1 py-0.5 text-accum">$1</code>')
      .replace(/_(.+?)_/g, "<i>$1</i>");

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("# ")) {
      out.push(<h1 key={i} className="mt-1 font-mono text-lg font-bold">{line.slice(2)}</h1>);
    } else if (line.startsWith("## ")) {
      out.push(<h2 key={i} className="mt-4 border-b border-border pb-1 font-mono text-sm font-semibold uppercase tracking-wider text-muted">{line.slice(3)}</h2>);
    } else if (line.startsWith("> ")) {
      out.push(<blockquote key={i} className="my-2 border-l-2 border-accum/60 bg-panel-2/50 py-2 pl-3 text-sm" dangerouslySetInnerHTML={{ __html: inline(line.slice(2)) }} />);
    } else if (line.startsWith("| ")) {
      const table: string[] = [];
      while (i < lines.length && lines[i].startsWith("|")) { table.push(lines[i]); i++; }
      i--;
      const rows = table.filter((r) => !/^\|[\s|:-]+\|$/.test(r));
      out.push(
        <div key={i} className="my-2 overflow-x-auto">
          <table className="w-full text-left font-mono text-[11px]">
            <tbody>
              {rows.map((r, ri) => {
                const cells = r.split("|").slice(1, -1).map((c) => c.trim());
                return (
                  <tr key={ri} className={ri === 0 ? "text-muted" : "border-t border-border/40"}>
                    {cells.map((c, ci) => (
                      <td key={ci} className="py-1 pr-3" dangerouslySetInnerHTML={{ __html: inline(c) }} />
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      );
    } else if (line.startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith("- ")) { items.push(lines[i].slice(2)); i++; }
      i--;
      out.push(
        <ul key={i} className="my-1 space-y-0.5 text-sm">
          {items.map((it, ii) => (
            <li key={ii} className="flex gap-2 text-text/90">
              <span className="text-muted">·</span>
              <span dangerouslySetInnerHTML={{ __html: inline(it) }} />
            </li>
          ))}
        </ul>
      );
    } else if (line.trim()) {
      out.push(<p key={i} className="my-1 text-sm text-text/90" dangerouslySetInnerHTML={{ __html: inline(line) }} />);
    }
    i++;
  }
  return out;
}

export function ReportPanel({ markdown }: { markdown: string }) {
  const [open, setOpen] = useState(true);
  return (
    <Card>
      <CardHeader>
        <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FileText size={13} /> Generated Report
          </CardTitle>
          <ChevronDown size={16} className={`text-muted transition-transform ${open ? "rotate-180" : ""}`} />
        </button>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="max-h-[520px] space-y-1 overflow-y-auto pr-2">{renderMarkdown(markdown)}</div>
        </CardContent>
      )}
    </Card>
  );
}
