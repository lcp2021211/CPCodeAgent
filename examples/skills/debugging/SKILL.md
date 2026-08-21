---
name: debugging
description: Diagnose and repair a reproducible code failure using evidence-first iterations.
requires-tools: [read_file, search_text, edit_file, run_command]
---

1. Reproduce or inspect the exact failure before editing.
2. Trace the failing behavior to the smallest relevant code path.
3. Make one focused change and preserve unrelated behavior.
4. Run the narrowest useful verification, then widen only if needed.
5. Report the cause, change, and verification evidence.

