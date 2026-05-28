import { cn } from "@/lib/utils";
import * as React from "react";

export function Badge({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5",
        "font-mono text-[10px] uppercase tracking-wider",
        className
      )}
      {...props}
    />
  );
}
