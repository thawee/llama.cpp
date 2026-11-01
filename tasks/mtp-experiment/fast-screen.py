#!/usr/bin/env python3
"""Candidate-only screens against saved IQ3 evidence, with service restoration."""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mtp_benchmark', HERE / 'benchmark.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@dataclass(frozen=True)
class Candidate:
    batch: int = 64
    draft_ubatch: int = 64
    iq3_min: int = 0
    bf16_min: int = 0
    bulk_hidden: bool = False
    long_prompt: bool = False
    q8_head: bool = False


CANDIDATES = {
    'mma1': Candidate(iq3_min=1, bf16_min=1),
    'mma7': Candidate(iq3_min=7, bf16_min=7),
    'batch128': Candidate(batch=128, draft_ubatch=16, long_prompt=True),
    'batch256': Candidate(batch=256, draft_ubatch=16, long_prompt=True),
    'bulk': Candidate(bulk_hidden=True),
    'q8head': Candidate(q8_head=True, long_prompt=True),
}


def model_path(candidate):
    if candidate.q8_head:
        return bench.ROOT / 'models/local/qwen3.8-27b-iq3-mtp-q8head.gguf'
    return bench.ROOT / 'models/local/qwen3.8-27b-iq3-mtp.gguf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', nargs='+', choices=CANDIDATES)
    parser.add_argument('--context', type=int, choices=[65536, 131072], default=65536)
    parser.add_argument('--runtime', type=Path, default=Path('/tmp/llama-memory-budget-build/bin'))
    parser.add_argument('--tag', default='screen')
    parser.add_argument('--short', action='store_true', help='Run only the two short saved workloads')
    parser.add_argument('--general', action='store_true', help='Also screen 128 general-thinking tokens')
    args = parser.parse_args()
    if not args.tag.replace('-', '').replace('_', '').isalnum():
        parser.error('Tag must contain only letters, digits, hyphens or underscores')
    bench.cpu.BINARY = args.runtime.resolve() / 'llama-server'
    bench.cpu.CONFIG = dict(bench.cpu.CONFIG, runtime=str(args.runtime.resolve()))
    bench.cpu.LIMIT = 22_000_000_000
    bench.REQUEST_TIMEOUT = 300
    bench.OUT = HERE / f'fast-{args.tag}-{args.context}.json'
    if bench.OUT.exists():
        raise FileExistsError(bench.OUT)
    baseline = HERE / 'results-65536-mtp5-q8_0-limit22-iq3-kapitsa.json'
    result = dict(status='running', trials=[], configured_context=args.context, draft_kv='q8_0',
                  kapitsa_cases=True, general_screen=args.general, limit_bytes=bench.cpu.LIMIT,
                  baseline=dict(path=str(baseline), sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                                configured_context=65536, context_matches=args.context == 65536),
                  runtime_sha256=hashlib.sha256(bench.cpu.BINARY.read_bytes()).hexdigest(),
                  methodology='Candidate-only single screens, same saved prompts/max96/temp0; unpaired comparison. No production baseline rerun.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 18092)) == 0:
            raise RuntimeError('Experiment port occupied')
    for name in args.candidates:
        path = model_path(CANDIDATES[name])
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        stopped = subprocess.run([str(bench.cpu.CONTROLLER), 'stop'], capture_output=True, text=True, timeout=30)
        if stopped.returncode:
            raise RuntimeError(stopped.stdout + stopped.stderr)
        result['system_memory_before'] = bench.system_memory()
        for name in args.candidates:
            candidate = CANDIDATES[name]
            bench.MODEL = model_path(candidate)
            env = {'LLAMA_MTP_BULK_HIDDEN': '1' if candidate.bulk_hidden else '0'}
            if candidate.iq3_min:
                env['GGML_METAL_MMA_IQ3_MIN_ROWS'] = str(candidate.iq3_min)
            if candidate.bf16_min:
                env['GGML_METAL_MMA_BF16_MIN_ROWS'] = str(candidate.bf16_min)
            result.update(model=str(bench.MODEL), batch=candidate.batch, ubatch=candidate.batch,
                          draft_ubatch=candidate.draft_ubatch, candidate_env=env,
                          case_names=['short_prose', 'short_code'] + (['long_prose', 'long_code'] if candidate.long_prompt and not args.short else []))
            bench.trial(result, name, 5, f'fast-{args.tag}-{args.context}', 1)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        restored = subprocess.run([str(bench.cpu.CONTROLLER), 'start'], capture_output=True, text=True, timeout=150)
        result['restored'] = dict(returncode=restored.returncode, output=restored.stdout + restored.stderr)
        bench.save(result)
        print('Production restored:', result['restored'], flush=True)
        if restored.returncode:
            raise RuntimeError('Production restoration failed')


if __name__ == '__main__':
    main()
