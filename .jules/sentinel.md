## 2025-04-18 - RCE Mitigations in Model Loading
**Vulnerability:** Insecure deserialization via `torch.load()` without `weights_only=True`, and arbitrary code execution via hardcoded `trust_remote_code=True` in Hugging Face model loading methods.
**Learning:** The application prioritized seamless loading of external models but unknowingly exposed users to serious remote code execution risks by implicitly trusting unknown model configurations and arbitrary pickle data.
**Prevention:** Default to strict, verifiable loading mechanisms. Use `weights_only=True` for torch checkpoints and require explicit opt-in (e.g., via CLI flag like `--trust-remote-code`) for features that execute remote code.
