## 2024-05-24 - Arbitrary Code Execution via HuggingFace Hub
**Vulnerability:** Loading models and tokenizers with `trust_remote_code=True`
**Learning:** HuggingFace `Auto*` classes can execute arbitrary Python code contained within the model repository if `trust_remote_code=True` is passed. This allows RCE if a user downloads a malicious model or if a legitimate model repository is compromised.
**Prevention:** Always use `trust_remote_code=False` (which is the default) unless there's a strong, verified need to execute remote code.
