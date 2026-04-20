## 2024-05-17 - Keyboard Accessibility in Python CLI Apps
**Learning:** Python's built-in `input()` function lacks keyboard accessibility out of the box on many platforms (no arrow key support for editing, no command history). This significantly degrades the UX for interactive CLI applications like chat loops.
**Action:** Always import the `readline` module (wrapped in a `try...except ImportError` block for cross-platform safety) in Python CLI scripts that use `input()` loops. It automatically intercepts `input()` to provide line editing and history capabilities.
