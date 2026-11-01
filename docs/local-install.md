# Private local installation

This optional macOS installation keeps the verified runtime outside the checkout. It does not modify the official `~/.local/llama` installation or `update-llama.sh`, and does not start a server during installation.

```sh
python3 scripts/install-local-runtime.py \
    --runtime build-m3-local/bin \
    --model models/local/qwen3.8-27b-iq3-mtp-q8head.gguf \
    --speculative draft-mtp
```

The installer copies `llama-server` and its dylibs into a content-addressed release under `~/.local/lib/llama-local/releases`, switches the `current` symlink atomically, and writes four managed wrappers to `~/.local/bin`. Add that directory to your shell's PATH if necessary. `--prefix` changes the runtime and wrapper location; configuration stays under the current HOME, which also permits isolated test installations. The model is referenced by absolute path and is not copied. Retain the model file and the Python interpreter used for installation. This verified build also depends on Homebrew OpenSSL 4 under `/opt/homebrew/opt/openssl@4`; copying the llama dylibs does not remove that external dependency.

```sh
llama-server --version      # Raw installed binary; accepts normal server flags
llama-local start          # Wait up to 120 seconds for the owned server
llama-local status         # Report launchd PID, listener ownership, readiness
llama-local logs           # Last 50 log lines
llama-pi                   # Start if necessary, then launch Pi with the local model
llama-copilot              # Start if necessary, then launch Copilot offline
llama-local stop           # Unload only this LaunchAgent
```

Client arguments are forwarded. `llama-pi --help` and `llama-copilot --help` do not start the service. The clients must already be installed. Pi keeps its existing extensions, settings, tools and project instructions. Copilot keeps its tool permissions and project instructions. The wrappers select the local provider and model; do not override these with cloud provider arguments.

The LaunchAgent is `local.llama.qwen27b`, with `RunAtLoad=false` and `KeepAlive=false`. It starts only on demand, does not restart on failure, and does not start at login. Simultaneous starts share a lock. A listening process is reused only when its PID matches this installed LaunchAgent and its health, model alias and 131072-token context pass checks. A foreign listener on port 8000 causes failure and is left untouched. Logs are in `~/.local/state/llama-local/server.log`.

The production server runs offline on `127.0.0.1:8000`, with alias `qwen3.8-27b-local`, 131072 context tokens, native MTP depth 5, one slot, 64-token batches and microbatches, five CPU threads, full Metal offload, flash attention, Q8_0 target/draft KV caches, and reasoning enabled. This MTP profile uses `--fit off`; the older `--memory-budget` estimator does not account for MTP. The short production-candidate screen peaked at 21.127 decimal GB RSS, which is not a process cap or a full-128K measurement. General-thinking sampling uses temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repetition penalty 1.0. Client request parameters can override server sampling defaults. For a foreground session, `LLAMA_LOCAL_SPECULATIVE=0 llama-local run` disables speculation and `LLAMA_LOCAL_MODEL=/absolute/model.gguf llama-local run` changes the model. The replacement must support the configured speculative mode. Stop the background server first. Environment overrides on a client command do not change an already loaded LaunchAgent's environment.

Configuration is in `~/.config/llama-local/config.json`; the generated LaunchAgent is in `~/Library/LaunchAgents/local.llama.qwen27b.plist`. Pi's `~/.pi/agent/models.json` receives the local provider and model. Existing providers, unrelated models and extra provider keys are preserved. Changed JSON files receive a `.bak` backup, with timestamped backups on subsequent changes. Identical reinstalls do not rewrite JSON. Unmanaged command collisions and invalid configuration fail before installation writes. Each file is replaced atomically; an unexpected filesystem error can leave a partial installation, so resolve it and rerun the installer.

Copilot uses the installed environment provider interface, disables online functionality with `COPILOT_OFFLINE=true`, and removes inherited provider credentials, headers, model IDs and common cloud API credentials before execution. A nonempty provider registry in `COPILOT_PROVIDERS_CONFIG` or under `COPILOT_HOME` (default `~/.copilot`) causes failure rather than silently selecting another provider. The configured prompt/output limits are 110592/8192 tokens. Pi uses `PI_OFFLINE=1` and its local dummy API key.

To verify the installation helpers without a model or launchd service:

```sh
python3 tests/test-local-install.py
```

These tests use temporary homes and mocked lifecycle calls. Real model loading, launchd ownership, streaming completions and tool calls require separate integration checks on the target machine.

## Updates and rollback

Stop the server before installing a different build, rerun the installer with the verified build directory, then start it again. Installation keeps existing content-addressed releases. The installer preserves the configured speculative profile unless `--speculative` is supplied. To restore IQ2, use the original `--model models/local/qwen3.8-27b-iq2_m.gguf` with `--speculative ngram-simple`; this restores its 18432 MiB inference estimate budget. To restore an older runtime, also pass `--runtime ~/.local/lib/llama-local/releases/<previous-release>`. Then run `llama-local start`. Configuration backups are beside the original JSON files; inspect a backup before restoring it so later unrelated edits are preserved. A reinstall does not stop or replace a running process.

The API is `http://127.0.0.1:8000/v1`; use `/health` to check readiness and the root URL for the web UI. `/v1` itself is an API prefix, not a web page. The server stays loaded after clients exit until `llama-local stop` is run.

## Verification on the local M3 Pro

The promoted IQ3 Q8-head profile passed launchd ownership, process-argument, 131072-context API and completed arithmetic-response checks. Evidence is in `tasks/production-iq3-q8head.json`. The following client tool checks were recorded earlier with IQ2.

Installation and reinstallation, startup ownership, three concurrent starts, repeated stop, and restart were checked on this machine. The server reports 131072 context tokens; streaming chat and a tool-result round trip passed. Pi and Copilot each read a temporary file through their tools and returned its contents. Those client smoke checks used only the file-read tool and disabled extra extensions/MCPs for isolation; they do not validate every installed plugin or a long coding session. Peak sampled server RSS during the client checks was 17.106 GB (decimal). This is a short-run observation, not a cap or proof of full-128K memory use.

Evidence is in `tasks/local-install-{api-result,client-results,lifecycle-result}.json` and the client logs. This deployment adds no inference speed optimization; see the README for the throughput target and measured baseline.
