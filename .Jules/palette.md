## 2024-05-24 - CLI Input UX
**Learning:** Python's built-in `input()` function can be seamlessly upgraded to support arrow keys (cursor navigation) and command history simply by importing the `readline` module on Unix-like systems. This transforms the interactive chat experience from rudimentary to fully-featured with zero structural code changes.
**Action:** Always import `readline` (wrapped in a `try...except ImportError` block for cross-platform compatibility) in any CLI application that uses an interactive `input()` loop.
