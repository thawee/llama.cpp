#!/usr/bin/env python3
"""Inspect head-only Q8 feasibility without modifying model files."""
import ctypes
import json
from pathlib import Path
import runpy
import shutil
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class InitParams(ctypes.Structure):
    _fields_ = [('no_alloc', ctypes.c_bool), ('ctx', ctypes.c_void_p)]


def parser_checks():
    lib = ctypes.CDLL(str(ROOT / 'build-m3-local/bin/libggml-base.dylib'))
    lib.gguf_init_from_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, InitParams]
    lib.gguf_init_from_buffer.restype = ctypes.c_void_p
    lib.gguf_free.argtypes = [ctypes.c_void_p]

    def tensor(name, shape, kind, offset):
        name = name.encode('ascii')
        return (struct.pack('<Q', len(name)) + name + struct.pack('<I', len(shape))
                + struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', kind, offset))

    def accepts(kind, offset):
        data = (b'GGUF' + struct.pack('<IQQ', 3, 2, 0)
                + tensor('output.weight', [64, 2], kind, 0)
                + tensor('output_norm.weight', [64], 0, offset))
        data += b'\0' * (-len(data) % 32) + b'\0' * 512
        buf = ctypes.create_string_buffer(data)
        ctx = lib.gguf_init_from_buffer(buf, len(data), InitParams(True, None))
        if ctx:
            lib.gguf_free(ctx)
        return bool(ctx)

    results = dict(bf16_contiguous=accepts(30, 256), q8_contiguous=accepts(8, 160),
                   q8_original_offsets_rejected=not accepts(8, 256))
    if not all(results.values()):
        raise RuntimeError(f'Native parser checks failed: {results}')
    return results


def main():
    checks = parser_checks()
    helpers = runpy.run_path(str(ROOT / 'tasks/mtp-experiment/prepare.py'))
    source = ROOT / 'models/local/qwen3.8-27b-iq3-mtp.gguf'
    with source.open('rb') as stream:
        fields, metadata, tensors, data_offset = helpers['parse'](stream.read(64 * 1024 * 1024))
    head = tensors[0]
    if (head['name'], head['shape'], head['kind'], head['offset']) != ('output.weight', [5120, 248320], 30, 0):
        raise ValueError('Unexpected source head layout')
    alignment = metadata.get('general.alignment', 32)
    head_elements = head['shape'][0] * head['shape'][1]
    old_bytes = head_elements * 2
    new_bytes = head_elements // 32 * 34
    reduction = ((old_bytes + alignment - 1) // alignment - (new_bytes + alignment - 1) // alignment) * alignment
    candidate_bytes = source.stat().st_size - reduction
    free = shutil.disk_usage(HERE).free
    provenance = json.loads((ROOT / 'tasks/mtp-experiment/recovered-iq3.json').read_text())
    report = dict(source=str(source), source_bytes=source.stat().st_size,
                  prior_publisher_verification=provenance, head_bf16_bytes=old_bytes,
                  head_q8_bytes=new_bytes, reduction_bytes=reduction,
                  unchanged_tensor_count=len(tensors) - 1,
                  unchanged_tail_bytes=source.stat().st_size - data_offset - old_bytes,
                  compact_candidate_bytes=candidate_bytes, free_bytes=free,
                  reserve_bytes=512 * 1024 * 1024,
                  extra_free_required=max(0, candidate_bytes + 512 * 1024 * 1024 - free),
                  native_parser_checks=checks,
                  candidate_created=False,
                  reason='First tensor shrinks; native GGUF parser rejects original subsequent offsets. APFS whole-file clone cannot avoid compacting the entire tail.')
    (HERE / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
