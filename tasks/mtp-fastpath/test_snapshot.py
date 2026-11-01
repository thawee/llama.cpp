#!/usr/bin/env python3
"""Compile actual MTP snapshot and acceptance bodies with a host-buffer fixture."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
source = (Path(sys.argv[1]) if len(sys.argv) > 1 else root / "common/speculative.cpp").read_text()
mtp = source[source.index("struct common_speculative_impl_draft_mtp") :]
start = mtp.index("        }\n\n", mtp.index("            if (!ok)")) + len("        }\n\n")
end = mtp.index("\n        return true;", start)
snapshot = mtp[start:end]
start = mtp.index("        if (seq_id < 0", mtp.index("void accept("))
end = mtp.index("\n    }", start)
accept = mtp[start:end]
harness = r'''#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
using llama_seq_id = int32_t;
struct context { std::vector<float> h; int row_calls = 0; int base_calls = 0; };
float * llama_get_embeddings_nextn(context * c) { ++c->base_calls; return c->h.data(); }
float * llama_get_embeddings_nextn_ith(context * c, int i) { ++c->row_calls; return c->h.data() + i * (c->h.size() / 8); }
struct fixture {
    int n_seq = 4, n_embd;
    bool bulk_hidden;
    context ctx;
    std::vector<int> i_batch_beg = {0, 1, -1, 4}, i_batch_end = {0, 3, -1, 7};
    std::vector<int> verify_h_rows = {0, 0, 0, 0};
    std::vector<std::vector<float>> verify_h = std::vector<std::vector<float>>(4);
    std::vector<std::vector<float>> pending_h;
    fixture(int width, bool bulk) : n_embd(width), bulk_hidden(bulk), pending_h(4, std::vector<float>(width, -1)) {
        ctx.h.resize(8 * width);
        for (size_t j = 0; j < ctx.h.size(); ++j) ctx.h[j] = float(j + 1);
    }
    void snapshot(bool shared) {
        context * ctx_tgt = &ctx;
        const size_t row_bytes = n_embd * sizeof(float);
        const float * h_tgt = shared ? nullptr : llama_get_embeddings_nextn(ctx_tgt);
SNAPSHOT
    }
    void accept(llama_seq_id seq_id, uint16_t n_accepted) {
ACCEPT
    }
};
int main() {
    for (int width : {1, 5120}) for (bool bulk : {false, true}) for (bool shared : {false, true}) {
        fixture f(width, bulk);
        f.snapshot(shared);
        const auto original = f.ctx.h;
        std::fill(f.ctx.h.begin(), f.ctx.h.end(), -9);
        assert(f.ctx.row_calls == (bulk ? 0 : 8));
        assert(f.ctx.base_calls == (!shared || bulk ? 1 : 0));
        for (int seq : {0, 1, 3}) {
            const int rows = f.i_batch_end[seq] - f.i_batch_beg[seq] + 1;
            assert(f.verify_h_rows[seq] == rows);
            for (int r = 0; r < rows; ++r) for (int j = 0; j < width; ++j)
                assert(f.verify_h[seq][r * width + j] == original[(f.i_batch_beg[seq] + r) * width + j]);
            for (int j = 0; j < width; ++j)
                assert(f.pending_h[seq][j] == original[f.i_batch_end[seq] * width + j]);
            for (uint16_t accepted : {0, 1, 2, 65535}) {
                f.accept(seq, accepted);
                const int row = std::min<int>(accepted, rows - 1) + f.i_batch_beg[seq];
                for (int j = 0; j < width; ++j) assert(f.pending_h[seq][j] == original[row * width + j]);
            }
        }
        f.accept(-1, 1); f.accept(4, 1); f.accept(2, 1);
        assert(f.verify_h[2].empty());
        assert(std::all_of(f.pending_h[2].begin(), f.pending_h[2].end(), [](float x) { return x == -1; }));
    }
    std::cout << "PASS: dense rows, per-sequence carryover, acceptance rollback, getter counts (8 configurations)\n";
}
'''
harness = harness.replace("SNAPSHOT", snapshot).replace("ACCEPT", accept)
out = root / "tasks/mtp-fastpath"
(out / "snapshot-test.cpp").write_text(harness)
subprocess.run(["c++", "-std=c++17", "-O1", "-g", "-fsanitize=undefined", str(out / "snapshot-test.cpp"), "-o", str(out / "snapshot-test")], check=True)
subprocess.run([str(out / "snapshot-test")], check=True, timeout=15)
