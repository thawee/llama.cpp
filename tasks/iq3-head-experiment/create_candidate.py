#!/usr/bin/env python3
"""Stream a head-only Q8_0 model; never rewrite the source."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HELPERS = runpy.run_path(str(ROOT / 'tasks/mtp-experiment/prepare.py'))
PREFLIGHT = runpy.run_path(str(HERE / 'preflight.py'))
LIB = ctypes.CDLL(str(ROOT / 'build-m3-local/bin/libggml-base.dylib'))
LIB.ggml_bf16_to_fp32_row.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
LIB.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
LIB.ggml_quantize_chunk.restype = ctypes.c_size_t
LIB.gguf_init_from_file.argtypes = [ctypes.c_char_p, PREFLIGHT['InitParams']]
LIB.gguf_init_from_file.restype = ctypes.c_void_p
LIB.gguf_free.argtypes = [ctypes.c_void_p]


def region_hash(path, start, length):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(start)
        while length:
            block = stream.read(min(length, 4 * 1024 * 1024))
            if not block:
                raise ValueError('Unexpected end of file during validation')
            digest.update(block)
            length -= len(block)
    return digest.hexdigest()


def create(source, destination, publisher_sha256):
    with source.open('rb') as stream:
        prefix = stream.read(min(source.stat().st_size, 64 * 1024 * 1024))
    fields, metadata, tensors, data_offset = HELPERS['parse'](prefix)
    head = tensors[0]
    if head['name'] != 'output.weight' or head['kind'] != 30 or head['offset'] != 0 or len(head['shape']) != 2:
        raise ValueError('Expected first tensor to be a BF16 output.weight matrix')
    columns, rows = head['shape']
    if columns % 32:
        raise ValueError('Q8_0 requires columns divisible by 32')
    alignment = metadata.get('general.alignment', 32)
    old_bytes = columns * rows * 2
    new_bytes = columns * rows // 32 * 34
    old_aligned = (old_bytes + alignment - 1) // alignment * alignment
    new_aligned = (new_bytes + alignment - 1) // alignment * alignment
    reduction = old_aligned - new_aligned
    if reduction <= 0 or len(tensors) < 2 or tensors[1]['offset'] != old_aligned:
        raise ValueError('Unexpected first tensor storage layout')
    expected_size = source.stat().st_size - reduction
    if shutil.disk_usage(destination.parent).free < expected_size + 512 * 1024 * 1024:
        raise OSError('Insufficient space for compact model plus 512 MiB reserve')
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + '.partial')
    header = bytearray(prefix[:data_offset])
    cursor = 24 + sum(len(raw) for raw in fields.values())
    for i, tensor in enumerate(tensors):
        cursor += len(tensor['raw'])
        struct.pack_into('<IQ', header, cursor - 12,
                         8 if i == 0 else tensor['kind'], tensor['offset'] - reduction if i else 0)
    source_digest = hashlib.sha256(prefix[:data_offset])
    tail_digest = hashlib.sha256()
    del prefix
    with source.open('rb') as src, partial.open('xb') as dst:
        dst.write(header)
        src.seek(data_offset)
        for row in range(0, rows, 128):
            count = min(128, rows - row)
            raw = src.read(columns * count * 2)
            if len(raw) != columns * count * 2:
                raise ValueError('Truncated BF16 head')
            source_digest.update(raw)
            raw_buffer = ctypes.create_string_buffer(raw)
            floats = (ctypes.c_float * (columns * count))()
            output = ctypes.create_string_buffer(columns * count // 32 * 34)
            LIB.ggml_bf16_to_fp32_row(raw_buffer, floats, columns * count)
            written = LIB.ggml_quantize_chunk(8, floats, output, 0, count, columns, None)
            if written != len(output):
                raise ValueError(f'Unexpected native quantizer size: {written}')
            dst.write(output.raw)
        source_digest.update(src.read(old_aligned - old_bytes))
        dst.write(b'\0' * (new_aligned - new_bytes))
        while block := src.read(4 * 1024 * 1024):
            source_digest.update(block)
            tail_digest.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    if source_digest.hexdigest() != publisher_sha256:
        raise ValueError('Complete source SHA256 differs from expected publisher digest; partial retained')
    if partial.stat().st_size != expected_size:
        raise ValueError('Candidate file size mismatch')
    tail_start = data_offset + new_aligned
    if region_hash(partial, tail_start, expected_size - tail_start) != tail_digest.hexdigest():
        raise ValueError('Candidate unchanged-tail digest mismatch')
    ctx = LIB.gguf_init_from_file(os.fsencode(partial), PREFLIGHT['InitParams'](True, None))
    if not ctx:
        raise ValueError('Native GGUF parser rejected candidate')
    LIB.gguf_free(ctx)
    with partial.open('rb') as stream:
        _, _, actual_tensors, actual_offset = HELPERS['parse'](stream.read(min(expected_size, 64 * 1024 * 1024)))
    if actual_offset != data_offset or len(actual_tensors) != len(tensors):
        raise ValueError('Candidate descriptor layout changed unexpectedly')
    for i, (old, new) in enumerate(zip(tensors, actual_tensors)):
        if (new['name'], new['shape'], new['kind'], new['offset']) != (
                old['name'], old['shape'], 8 if i == 0 else old['kind'], old['offset'] - reduction if i else 0):
            raise ValueError(f'Unexpected descriptor change: {old["name"]}')
    os.link(partial, destination)
    partial.unlink()
    return dict(source=str(source), candidate=str(destination), source_sha256=source_digest.hexdigest(),
                publisher_sha256_verified=True, candidate_bytes=expected_size,
                candidate_sha256=region_hash(destination, 0, expected_size),
                unchanged_tail_sha256=tail_digest.hexdigest(), unchanged_tensors=len(tensors) - 1,
                unchanged_tail_bytes=expected_size - tail_start, native_gguf_parser_valid=True,
                head=dict(name=head['name'], shape=head['shape'], original_type='BF16', candidate_type='Q8_0',
                          original_bytes=old_bytes, candidate_bytes=new_bytes),
                quality_status='Changed target output precision; runtime and quality screening pending')


def self_test():
    PREFLIGHT['parser_checks']()
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        source = Path(temp) / 'source.gguf'
        candidate = Path(temp) / 'candidate.gguf'
        raw = bytearray(b'GGUF' + struct.pack('<IQQ', 3, 3, 0))
        tensors = [('output.weight', [64, 2], 30, 0), ('output_norm.weight', [64], 0, 256),
                   ('blk.64.nextn.shared_head.weight', [64, 1], 30, 512)]
        for name, shape, kind, offset in tensors:
            name = name.encode('ascii')
            raw += struct.pack('<Q', len(name)) + name + struct.pack('<I', len(shape))
            raw += struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', kind, offset)
        raw += b'\0' * (-len(raw) % 32)
        data_offset = len(raw)
        raw += struct.pack('<128H', *([0x3F80, 0xBF00] * 64)) + bytes(range(256)) + bytes(range(128))
        source.write_bytes(raw)
        expected_hash = hashlib.sha256(raw).hexdigest()
        report = create(source, candidate, expected_hash)
        after = candidate.read_bytes()
        assert source.read_bytes() == raw
        assert after[data_offset + 160:] == raw[data_offset + 256:]
        assert after[data_offset:data_offset + 136] != raw[data_offset:data_offset + 136]
        scale = struct.unpack_from('<e', after, data_offset)[0]
        values = struct.unpack_from('<32b', after, data_offset + 2)
        assert max(abs(scale * value - expected) for value, expected in zip(values, [1.0, -0.5] * 16)) < 0.01
        assert report['unchanged_tensors'] == 2
        try:
            create(source, candidate, expected_hash)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing destination was accepted')
        original_link = os.link
        try:
            def raced_link(src, dst):
                Path(dst).write_bytes(b'concurrent artifact')
                return original_link(src, dst)
            os.link = raced_link
            concurrent = Path(temp) / 'concurrent.gguf'
            try:
                create(source, concurrent, expected_hash)
            except FileExistsError:
                assert concurrent.read_bytes() == b'concurrent artifact'
            else:
                raise AssertionError('Concurrent destination was replaced')
        finally:
            os.link = original_link
    print('Self-test: native quantization error <0.01, original preservation, norm/MTP byte preservation, parser and overwrite refusal passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    provenance = json.loads((ROOT / 'tasks/mtp-experiment/recovered-iq3.json').read_text())
    report = create(Path(provenance['model']), HERE / 'qwen3.8-27b-iq3-mtp-q8-head.gguf', provenance['sha256'])
    report['provenance'] = provenance
    (HERE / 'candidate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
