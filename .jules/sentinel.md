
## 2024-05-24 - Default Arbitrary Code Execution in Model Loading
**Vulnerability:** Found `torch.load` missing `weights_only=True` leading to insecure deserialization risk, and `trust_remote_code=True` hardcoded in Hugging Face loading mechanisms, which permits automatic execution of untrusted remote code (ACE/RCE) simply by running the script against a malicious model repository.
**Learning:** Hugging Face and PyTorch model loading APIs default to allowing remote code execution if explicitly enabled without user intervention. Opting into such trust automatically poses a critical risk.
**Prevention:** Always enforce `weights_only=True` for `torch.load` and ensure that `trust_remote_code` defaults to `False`. It should only be enabled via a direct, explicit opt-in argument provided by the user (e.g., `--trust-remote-code`).
