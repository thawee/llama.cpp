#!/usr/bin/env python3
"""Build an isolated split GGUF with unchanged target tensors and a pinned MTP head."""
import ctypes
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET = ROOT / 'models/local/qwen3.8-27b-iq2_m.gguf'
FIRST = TARGET.with_name('qwen3.8-27b-iq2-mtp-00001-of-00002.gguf')
SECOND = TARGET.with_name('qwen3.8-27b-iq2-mtp-00002-of-00002.gguf')
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def kv(key, kind, value):
    return string(key) + struct.pack('<I', kind) + (string(value) if kind == 8 else struct.pack('<' + FORMATS[kind], value))


def parse(data):
    f = io.BytesIO(data)
    def scalar(fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
    def text():
        return f.read(scalar('Q')).decode()
    def value(kind, keep=True):
        if kind == 8:
            if keep:
                return text()
            f.seek(scalar('Q'), 1)
            return None
        if kind == 9:
            element, count = scalar('I'), scalar('Q')
            if element in FORMATS:
                f.seek(count * struct.calcsize(FORMATS[element]), 1)
            else:
                for _ in range(count):
                    value(element, False)
            return None
        return scalar(FORMATS[kind])
    if f.read(4) != b'GGUF':
        raise ValueError('Invalid GGUF magic')
    version, count, keys = scalar('I'), scalar('Q'), scalar('Q')
    if version != 3:
        raise ValueError('Expected GGUF v3')
    fields, metadata, tensors = {}, {}, []
    for _ in range(keys):
        start = f.tell()
        key, kind = text(), scalar('I')
        metadata[key] = value(kind)
        fields[key] = data[start:f.tell()]
    for _ in range(count):
        start = f.tell()
        name, dimensions = text(), scalar('I')
        shape = [scalar('Q') for _ in range(dimensions)]
        kind, offset = scalar('I'), scalar('Q')
        tensors.append(dict(name=name, shape=shape, kind=kind, offset=offset, raw=data[start:f.tell()]))
    alignment = metadata.get('general.alignment', 32)
    return fields, metadata, tensors, math.ceil(f.tell() / alignment) * alignment


def header(fields, tensors, fixed_size=None):
    fields = dict(fields)
    body = b''.join(t['raw'] for t in tensors)
    if fixed_size is not None:
        base_size = 24 + sum(map(len, fields.values())) + len(body)
        padding = fixed_size - base_size - len(kv('local.header_padding', 8, ''))
        if padding < 0:
            raise ValueError('Insufficient header space for an unchanged data offset')
        fields['local.header_padding'] = kv('local.header_padding', 8, ' ' * padding)
    result = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(fields)) + b''.join(fields.values()) + body
    result += b'\0' * (-len(result) % 32)
    if fixed_size is not None and len(result) != fixed_size:
        raise ValueError('Header changed the target data offset')
    return result


def digest(path, start=0):
    h = hashlib.sha256()
    with path.open('rb') as f:
        f.seek(start)
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    if '--self-test' in sys.argv:
        fields = {'general.architecture': kv('general.architecture', 8, 'qwen35')}
        raw = string('test.weight') + struct.pack('<IQQIQ', 2, 32, 2, 8, 0)
        result = header(fields, [{'raw': raw}], fixed_size=256)
        _, meta, tensors, offset = parse(result)
        assert meta['general.architecture'] == 'qwen35' and offset == 256
        assert tensors[0]['shape'] == [32, 2] and tensors[0]['offset'] == 0
        print('GGUF header round-trip and fixed-offset checks passed')
        return
    if FIRST.exists() or SECOND.exists():
        raise FileExistsError('Experiment outputs already exist; refusing overwrite')
    with TARGET.open('rb') as f:
        fields, meta, trunk, offset = parse(f.read(32 * 1024 * 1024))
    _, source_meta, source_tensors, source_offset = parse((HERE / 'source-header.bin').read_bytes())
    for key in ['general.architecture', 'qwen35.embedding_length', 'qwen35.attention.head_count',
                'qwen35.attention.head_count_kv', 'qwen35.feed_forward_length']:
        if meta[key] != source_meta[key]:
            raise ValueError('Incompatible MTP source: ' + key)
    assert meta['qwen35.block_count'] == 64 and source_meta['qwen35.nextn_predict_layers'] == 1
    mtp = [t for t in source_tensors if t['name'].startswith('blk.64.')]
    assert len(mtp) == 15 and all(t['kind'] in (0, 8) for t in mtp)
    start = min(t['offset'] for t in mtp)
    def nbytes(t):
        n = math.prod(t['shape'])
        return n * 4 if t['kind'] == 0 else n // 32 * 34
    end = max(t['offset'] + nbytes(t) for t in mtp)
    for t in source_tensors:
        assert t in mtp or not (start <= t['offset'] < end)
    source = json.loads((HERE / 'source.json').read_text())
    lo, hi = source_offset + start, source_offset + end - 1
    payload = HERE / 'mtp-q8.payload'
    request = urllib.request.Request(source['url'] + '?mtp_payload=1', headers={'Range': f'bytes={lo}-{hi}'})
    print('Downloading only MTP tensor span:', end - start, 'bytes', flush=True)
    with urllib.request.urlopen(request, timeout=90) as response, payload.open('xb') as out:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/14532048416':
            raise ValueError('Remote server did not honor exact pinned range')
        remaining = end - start
        while remaining:
            block = response.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise EOFError('Truncated MTP download')
            out.write(block)
            remaining -= len(block)
    fields = {k: v for k, v in fields.items() if not k.startswith('quantize.imatrix.')}
    fields.update({'qwen35.block_count': kv('qwen35.block_count', 4, 65),
                   'qwen35.nextn_predict_layers': kv('qwen35.nextn_predict_layers', 4, 1),
                   'split.count': kv('split.count', 2, 2),
                   'split.tensors.count': kv('split.tensors.count', 5, len(trunk) + len(mtp))})
    fields['split.no'] = kv('split.no', 2, 0)
    first_header = header(fields, trunk, fixed_size=offset)
    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    if clonefile(str(TARGET).encode(), str(FIRST).encode(), 0):
        raise OSError(ctypes.get_errno(), 'APFS clonefile failed')
    with FIRST.open('r+b') as f:
        f.write(first_header)
    fields['split.no'] = kv('split.no', 2, 1)
    for t in mtp:
        t['raw'] = t['raw'][:-8] + struct.pack('<Q', t['offset'] - start)
    with SECOND.open('xb') as f, payload.open('rb') as source_payload:
        f.write(header(fields, mtp))
        while block := source_payload.read(4 * 1024 * 1024):
            f.write(block)
    print('Checking complete target tensor payload identity', flush=True)
    original_hash = digest(TARGET, offset)
    assert original_hash == digest(FIRST, offset)
    record = {'target': str(TARGET), 'first': str(FIRST), 'second': str(SECOND),
              'target_data_offset': offset, 'trunk_tensors': len(trunk), 'mtp_tensors': len(mtp),
              'mtp_bytes': end - start, 'mtp_sha256': digest(payload),
              'trunk_payload_sha256': original_hash, 'trunk_payload_unchanged': True,
              'source_revision': source['revision'], 'range': [lo, hi]}
    (HERE / 'prepared.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Prepared isolated split GGUF; original model unchanged', flush=True)


if __name__ == '__main__':
    main()
