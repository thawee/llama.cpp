# Native MTP bulk host snapshot candidate

Enable only with `LLAMA_MTP_BULK_HIDDEN=1`; unset, `0`, and other values retain the prior path. Source is final and ready for integration build. No commits, model runs, or server lifecycle operations performed.

## Scope and behavior

Only `common/speculative.cpp` changed. `candidate.patch` is relative to the saved initial `speculative.before.cpp`; `preexisting.patch` records the edits already present before this candidate. Existing backend sampling and adaptive-backoff edits remain intact.

MTP target hidden states are `[raw target batch row][n_embd]` float rows. The target constructor enables unmasked rows. Existing process code assumes contiguous per-sequence batch spans. The candidate retains that assumption, copies each span to the existing owned `verify_h`, and retains the pending-row and accept-index selection unchanged.

Unshared-memory MTP reuses the target base pointer acquired with synchronization before draft catch-up. Shared-memory MTP acquires that pointer with synchronization once during snapshotting. Default path continues using the existing row getter. No synchronization required before consuming device output was deleted. Target output rows remain owned by the target context throughout draft catch-up; the draft context does not decode the target context.

## Evidence

- `src/llama-context.cpp:1125`: base getter performs output reordering and returns the host nextn buffer.
- `src/llama-context.cpp:1141`: unmasked nextn rows are dense raw token rows.
- `src/llama-context.cpp:2151`: target extraction uses all unmasked ubatch rows and raw offsets.
- `src/llama-context.cpp:4344`: each public nextn getter synchronizes context before access.
- `src/llama-graph.cpp:155`: draft state input requires a host `ubatch->embd_state`, then calls `ggml_backend_tensor_set`.

## GPU-resident blocker

The staging API exposes only host float pointers. The target graph already schedules async device-to-host nextn extraction; draft graph inputs upload host state through tensor_set. A direct handoff requires a tensor/buffer lifetime contract across target and draft graphs/schedulers, compatible backend placement, device-copy ordering, and rollback row selection. Changing only this file cannot supply those contracts. Bulk snapshotting reduces getter/synchronize invocations and host memcpy invocations, not transferred bytes, retained state, or required initial synchronization.

## Verification and remaining limits

`python3 tasks/mtp-fastpath/test_snapshot.py` extracts the actual snapshot and accept bodies into a fixture, compiles with UBSan, and checks widths 1 and 5120, 1/3/4-row sequences, absent sequences, shared/nonshared pointer acquisition, default/opt-in behavior, acceptance 0/1/2/65535, and owned snapshot lifetime after target host-buffer overwrite. The saved pre-candidate source fails the expected optimized getter-count assertion. The candidate passes all 8 configurations. This is a focused host-level regression, not a real decoder integration test.

Production translation unit compiled using the existing build-bench release flags into isolated `speculative.cpp.o`: exit 0. `git diff --check -- common/speculative.cpp`: clean. ASan could not initialize on this host; its sampled stack spins in shadow-memory/dyld startup before main. Those fixture processes were stopped; UBSan works. Logs and sample are retained here.

No throughput/memory improvements are claimed. Main agent owns real inference and short workload measurement using saved compatible baselines. GPU races, backend lifetime interactions, and actual model acceptance still need real inference validation.
