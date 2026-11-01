# Local installation review

Scope: README, installer, controller, isolated tests and installation guide. Existing inference changes were preserved and are outside this deployment diff.

Reviewed correctness, simplicity, ownership, credentials, and runtime behavior using the repository review checklist. No core inference or backend edits were added. Deployment uses content-addressed releases, a launchd-owned process, a serialized start/stop lock, listener ownership checks, and atomic configuration merges with backups. Shell wrappers quote installed paths; copied runtime hashes match the verified build. Copilot uses the installed provider interface and offline mode; Pi uses its local provider and offline startup setting. Unknown provider registries fail explicitly.

Verification: 13 tests passed, plus real launchd and client checks recorded in local-install result artifacts. Final runtime is healthy after restart and arithmetic returned 42. No unresolved deployment correctness findings. Remaining limits: external Python/OpenSSL/model paths must remain available; per-file atomic installation is not a multi-file transaction; full-128K occupancy and long sessions are not verified. Client smoke tests isolate extensions/MCPs; production wrappers preserve user tool permissions and instructions.

Work remains uncommitted on local/memory-budget. No branch integration, publication, or cleanup requested.
