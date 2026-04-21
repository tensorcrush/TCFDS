## 2024-04-21 - Python CLI Input Accessibility
**Learning:** Standard Python `input()` function lacks keyboard accessibility (up/down history, left/right arrow key line editing). This forces users to delete their entire line to fix a typo and prevents them from easily repeating commands. Importing the built-in `readline` module magically adds this functionality to `input()` on Unix systems.
**Action:** Always import `readline` (wrapped in a try-except block for cross-platform compatibility) when building CLI applications in Python that use `input()` loops.
