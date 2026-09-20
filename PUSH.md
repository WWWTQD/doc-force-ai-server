PUSH.md

Purpose
-------
This file lists sensitive files, folders and patterns that MUST NOT be pushed to remote repositories. Keep this file updated and consult it before committing or pushing.

Usage
-----
- Before creating a commit or pushing, run:

```powershell
cd C:\Users\imdrx\IdeaProjects\doc-force-ai-server
git status --porcelain
```

- Only `git add` the specific files you want to push (avoid using `git add .` if you are unsure).

- To add this file only:

```powershell
git add PUSH.md
git commit -m "docs: add PUSH.md (sensitive files list)"
git push -u origin manoi
```

Sensitive files / patterns (DO NOT PUSH)
---------------------------------------
These are examples and should be kept out of Git.

- Environment and secrets
  - .env
  - .env.local
  - .secrets
  - config/*.secret

- Virtual environments
  - .venv/
  - venv/

- Local data and uploads
  - data/input/
  - data/output/
  - uploads/
  - outputs/
  - firmware/

- Local AI / debug files
  - LOCAL_AI_STATUS.md
  - any local_notes/*.md

- Model weights and large binaries
  - models/
  - *.pth
  - *.pt
  - *.onnx
  - *.safetensors

- IDE/editor config that may contain local paths or credentials
  - .idea/
  - .vscode/

- Logs and caches
  - *.log
  - __pycache__/
  - *.pyc

- Private keys and credentials
  - id_rsa
  - id_rsa.pub (if private keys sneak in)
  - *.pem
  - *.key

- Platform-specific files
  - .DS_Store
  - Thumbs.db

Recommendations
---------------
- Keep `.gitignore` updated to match the list above.
- Use tools like `git-secrets` or pre-commit hooks to block accidental commits of secrets.
- If you accidentally commit a secret, remove it from history using `git filter-repo` or `bfg` and rotate the secret.

Notes for CI / Automation
-------------------------
- CI should load secrets via environment variables or secret stores (GitHub Actions secrets, Vault, etc.) not from repository files.

Contact
-------
If you are unsure whether a file should be pushed, ask the repository owner or check this file before pushing.

