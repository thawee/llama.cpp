MTP host hidden-row candidate

- [x] Trace dense target rows, synchronization, and rollback lifetime.
- [x] Add regression harness against extracted production snapshot/accept blocks.
- [x] Add opt-in contiguous host snapshot; preserve existing default.
- [x] Run host harness and compile common library.
- [x] Save patch, evidence, architecture limits for main GPU owner.

Checkpoint: this candidate reduces host getter/sync calls and memcpy calls, not device readback volume; main owns short GPU measurements using saved baselines.
