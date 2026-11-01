#!/usr/bin/env python3
"""Bounded native MTP screen with general-thinking sampling and service restoration."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('cpu_checks', ROOT / 'tasks/cpu-core-benchmark.py')
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
OUT = HERE / 'results.json'
MODEL = ROOT / 'models/local/qwen3.8-27b-iq2-mtp-00001-of-00002.gguf'
PROMPTS = {
    'summary': 'Summarize this community update in plain language, separating confirmed facts from proposals. '
               'The library opened on Monday. It has 120 seats and is closed on Sundays. '
               'The council is considering evening opening, but has not approved funding. '
               'Volunteers proposed a weekly reading club; no date has been chosen. '
               'Explain what residents can use now and what is still undecided.',
    'planning': 'Help me plan a relaxed Saturday with two adults and a ten-year-old. '
                'We have a total budget of 80 dollars, no car, and a park and public library within walking distance. '
                'Rain is possible. Give a practical schedule with a rainy-day alternative and a simple budget. '
                'Do not invent local opening hours or ticket prices; label estimates and things to check.'
}
SAMPLING = dict(temperature=1.0, top_p=.95, top_k=20, min_p=0., presence_penalty=0., repeat_penalty=1.)
REQUEST_TIMEOUT = 180


def longer_prompt(records):
    entries = '\n'.join(f'Entry {i}: The community library has 120 seats and closes on Sundays. '
                         'Evening opening is proposed, funding is undecided. A reading club has no confirmed date.'
                         for i in range(records))
    return entries + '\nSummarize the confirmed services and the undecided proposals. Do not invent details.'


def system_memory():
    return {key: subprocess.check_output(command, text=True).strip() for key, command in {
        'swap': ['sysctl', 'vm.swapusage'], 'vm': ['vm_stat']}.items()}


def kapitsa_cases():
    system = 'Reference constants. Use these only if relevant to the question.\n' + '\n'.join(
        f'constant_{i:04d} = {1000+i}' for i in range(350))
    prose = 'Explain how a seed grows into a tree. Describe each stage in clear numbered paragraphs, with at least 250 words.'
    code = 'Write only Python code for merge_sort(values), with a detailed docstring, the merge helper, and five assert-based example tests. Do not use sorted().'
    return [('short_prose', prose, ''), ('short_code', code, ''),
            ('long_prose', prose, system), ('long_code', code, system)]


def kapitsa_completion(question, system):
    messages = ([dict(role='system', content=system)] if system else []) + [dict(role='user', content=question)]
    body = dict(model='local', messages=messages, temperature=0, max_tokens=96,
                reasoning_effort='none', chat_template_kwargs={'enable_thinking': False})
    start = time.monotonic()
    response = cpu.runtime.request('/v1/chat/completions', body, timeout=REQUEST_TIMEOUT)
    if not response['choices'][0]['message'].get('content', '').strip() or response['usage']['completion_tokens'] < 80:
        raise RuntimeError('Kapitsa workload returned empty or unexpectedly short output')
    return dict(content=response['choices'][0]['message']['content'], timings=response['timings'],
                wall_seconds=time.monotonic() - start, settings=body, usage=response['usage'])


def save(result):
    temporary = OUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(OUT)


def completion(prompt, count=128, thinking=True, overrides=None):
    formatted = cpu.runtime.request('/apply-template', {'messages': [{'role': 'user', 'content': prompt}],
        'chat_template_kwargs': {'enable_thinking': thinking}})['prompt']
    settings = dict(SAMPLING, seed=1234, n_predict=count, ignore_eos=True, cache_prompt=False, stream=False)
    settings.update(overrides or {})
    start = time.monotonic()
    response = cpu.runtime.request('/completion', dict(settings, prompt=formatted), timeout=REQUEST_TIMEOUT)
    response['wall_seconds'] = time.monotonic() - start
    if response['timings']['predicted_n'] != count or response['timings']['prompt_n'] <= 0:
        raise RuntimeError('Unexpected measured token work')
    return {k: response[k] for k in ['content', 'timings', 'wall_seconds']} | {'settings': settings}


def watch(process, done, record):
    while not done.is_set() and process.poll() is None:
        try:
            value = cpu.usage(process.pid)
            pressure = int(subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True))
            record['samples'].append(dict(time=time.time(), rss=value.resident, footprint=value.footprint, pressure=pressure))
            if max(value.resident, value.footprint) > cpu.LIMIT or pressure >= 4:
                raise RuntimeError(f'{cpu.LIMIT / 1e9:g} decimal GB ceiling or critical memory pressure')
        except Exception as error:
            if process.poll() is None:
                record['events'].append(str(error))
                os.killpg(process.pid, signal.SIGTERM)
            break
        done.wait(.5)
    if record['events'] and process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            record['events'].append('Watchdog escalated to SIGKILL after five seconds')


def trial(result, name, depth, phase, repetition):
    context = result['configured_context']
    args = [str(cpu.BINARY), '-m', str(MODEL if depth else cpu.CONFIG['model']),
            '--host', '127.0.0.1', '--port', '18092', '--offline', '-ngl', '99', '--fit', 'off',
            '-np', '1', '-c', str(context), '-b', str(result.get('batch', 64)),
            '-ub', str(result.get('ubatch', 64)), '-fa', 'on',
            '-ctk', 'q8_0', '-ctv', 'q8_0', '-t', '5', '-tb', '5', '--reasoning', 'on',
            '--temp', '1', '--top-p', '.95', '--top-k', '20', '--min-p', '0',
            '--presence-penalty', '0', '--repeat-penalty', '1']
    if result.get('kapitsa_cases'):
        args[args.index('--reasoning') + 1] = 'off'
        args[args.index('--temp') + 1] = '0'
    if depth:
        args += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(depth)]
        if result.get('draft_ubatch'):
            args += ['--spec-draft-n-ubatch', str(result['draft_ubatch'])]
        if result.get('draft_kv'):
            args += ['--spec-draft-type-k', result['draft_kv'], '--spec-draft-type-v', result['draft_kv']]
    else:
        args += ['--memory-budget', '18432', '--spec-type', 'ngram-simple',
                 '--spec-ngram-simple-size-n', '3', '--spec-ngram-simple-size-m', '3']
    record = dict(name=name, depth=depth, phase=phase, repetition=repetition, argv=args,
                  samples=[], events=[], log=str(HERE / f'{phase}-{name}-{repetition}.log'), measurements={})
    result['trials'].append(record)
    save(result)
    print('Starting', phase, repetition, name, flush=True)
    env = dict(os.environ, DYLD_LIBRARY_PATH=cpu.CONFIG['runtime'])
    for key in list(env):
        if key.startswith(('LLAMA_ARG_', 'GGML_METAL_')):
            env.pop(key)
    env.update(result.get('candidate_env', {}))
    record['candidate_env'] = result.get('candidate_env', {})
    done = threading.Event()
    process = None
    watcher = None
    try:
        with open(record['log'], 'w') as log:
            process = subprocess.Popen(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            watcher = threading.Thread(target=watch, args=(process, done, record), daemon=True)
            watcher.start()
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError('Server exited during startup; inspect trial log')
                try:
                    if cpu.runtime.request('/health', timeout=1).get('status') == 'ok':
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise RuntimeError('Server startup timed out')
            slots = cpu.runtime.request('/slots')
            if len(slots) != 1 or slots[0]['n_ctx'] != context:
                raise RuntimeError('Context changed')
            record['warmup'] = completion('Reply with ready.', 8, False)
            prompts = {'long80': longer_prompt(80), 'long320': longer_prompt(320)} if result.get('long_prompts') else PROMPTS
            cases = kapitsa_cases() if result.get('kapitsa_cases') else [(k, v, '') for k, v in prompts.items()]
            if result.get('case_names'):
                cases = [case for case in cases if case[0] in result['case_names']]
            for label, prompt, system in cases:
                print('Processing', label, flush=True)
                record['measurements'][label] = kapitsa_completion(prompt, system) if result.get('kapitsa_cases') else completion(prompt)
                print('Measured', label, record['measurements'][label]['timings'], flush=True)
                record.setdefault('system_memory_after_workload', {})[label] = system_memory()
                save(result)
            if result.get('general_screen'):
                record['general_measurement'] = completion(PROMPTS['summary'], count=128, thinking=True)
                save(result)
            check = cpu.runtime.request('/v1/chat/completions', {'model': 'local',
                'messages': [{'role': 'user', 'content': 'What is 17+25? Reply with only the number.'}],
                'temperature': 0, 'max_tokens': 256, 'reasoning_budget': 64}, timeout=120)
            record['quality_check'] = check
            if '42' not in (check['choices'][0]['message'].get('content') or ''):
                raise RuntimeError('Arithmetic response failed')
            if record['events']:
                raise RuntimeError('Memory watchdog event')
            record['passed'] = True
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        done.set()
        if watcher:
            watcher.join(timeout=5)
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                record['exit_code'] = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                record['exit_code'] = process.wait(timeout=10)
        save(result)
    print('Completed', name, {k: round(v['timings']['predicted_per_second'], 3)
                              for k, v in record['measurements'].items()}, flush=True)


def main():
    global OUT, REQUEST_TIMEOUT, MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--confirm-depth', type=int, choices=[1, 2, 3, 5])
    parser.add_argument('--depth', type=int, choices=[1, 2, 3, 5])
    parser.add_argument('--context', type=int, choices=[65536, 131072], default=131072)
    parser.add_argument('--draft-kv', choices=['f16', 'q8_0'])
    parser.add_argument('--memory-limit-gb', type=int, choices=[20, 22], default=20)
    parser.add_argument('--long-prompts', action='store_true')
    parser.add_argument('--iq3', action='store_true')
    parser.add_argument('--kapitsa-cases', action='store_true')
    args = parser.parse_args()
    if (args.draft_kv or args.long_prompts or args.iq3 or args.kapitsa_cases or args.memory_limit_gb != 20) and not args.depth:
        parser.error('Candidate options require --depth to preserve separate evidence')
    if args.long_prompts and args.kapitsa_cases:
        parser.error('Select one workload profile')
    if args.self_test:
        assert set(PROMPTS) == {'summary', 'planning'} and SAMPLING['temperature'] == 1.0
        assert cpu.LIMIT == 20_000_000_000 and MODEL.is_file()
        assert len(longer_prompt(320)) > len(longer_prompt(80)) * 3
        assert len(kapitsa_cases()) == 4 and kapitsa_cases()[-1][2].count('constant_') == 350
        print('Harness configuration checks passed; no service changed')
        return
    cpu.LIMIT = args.memory_limit_gb * 1_000_000_000
    REQUEST_TIMEOUT = 1800 if args.long_prompts or args.kapitsa_cases else 180
    if args.iq3:
        MODEL = MODEL.with_name('qwen3.8-27b-iq3-mtp.gguf')
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    experiment_suffix = (f'-limit{args.memory_limit_gb}' if args.memory_limit_gb != 20 else '') + ('-long' if args.long_prompts else '')
    experiment_suffix += ('-iq3' if args.iq3 else '') + ('-kapitsa' if args.kapitsa_cases else '')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 18092)) == 0:
            raise RuntimeError('Experiment port is occupied')
    if args.depth:
        suffix = f'-{args.draft_kv}' if args.draft_kv else ''
        suffix += experiment_suffix
        OUT = HERE / f'results-{args.context}-mtp{args.depth}{suffix}.json'
        if OUT.exists():
            raise FileExistsError('Single-depth result exists; refusing to overwrite evidence')
    result = json.loads(OUT.read_text()) if args.confirm_depth else dict(trials=[])
    baseline_file = HERE / 'results.json'
    if baseline_file.exists():
        cached = json.loads(baseline_file.read_text())
        result['saved_baseline'] = {'path': str(baseline_file), 'context': cached['configured_context'],
            'trials': [t for t in cached['trials'] if t['name'] == 'baseline' and t.get('passed')]}
    result.update(status='running', limit_bytes=cpu.LIMIT, configured_context=args.context,
                  draft_kv=args.draft_kv,
                  long_prompts=args.long_prompts, system_memory_before=system_memory(),
                  kapitsa_cases=args.kapitsa_cases, model=str(MODEL),
                  runtime_sha256=hashlib.sha256(cpu.BINARY.read_bytes()).hexdigest(),
                  methodology='General thinking, temperature1, fixed128 tokens including reasoning; synthetic longer prompts' if args.long_prompts else
                              'General thinking, temperature1, fixed128 tokens including reasoning; short prompts, sampled outputs may differ.')
    if args.kapitsa_cases:
        result['methodology'] = 'Saved Kapitsa four-case workload; non-thinking, temperature zero, max96 tokens; no baseline rerun.'
    try:
        stopped = subprocess.run([str(cpu.CONTROLLER), 'stop'], capture_output=True, text=True, timeout=30)
        if stopped.returncode:
            raise RuntimeError(stopped.stdout + stopped.stderr)
        if args.confirm_depth:
            for repetition in range(1, 6):
                order = [(f'mtp{args.confirm_depth}', args.confirm_depth)]
                for name, depth in order:
                    trial(result, name, depth, 'confirmation', repetition)
        elif args.depth:
            phase = f'context{args.context}' + (f'-{args.draft_kv}' if args.draft_kv else '')
            phase += experiment_suffix
            trial(result, f'mtp{args.depth}', args.depth, phase, 1)
        else:
            if not result.get('saved_baseline', {}).get('trials'):
                raise RuntimeError('Saved baseline required; do not rerun production baseline')
            for name, depth in [('mtp1', 1), ('mtp2', 2), ('mtp3', 3), ('mtp5', 5)]:
                trial(result, name, depth, 'screen', 1)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            result['system_memory_after'] = system_memory()
        except subprocess.CalledProcessError as error:
            result['system_memory_error'] = repr(error)
        restored = subprocess.run([str(cpu.CONTROLLER), 'start'], capture_output=True, text=True, timeout=150)
        result['restored'] = dict(returncode=restored.returncode, output=restored.stdout + restored.stderr)
        save(result)
        print('Shared server restored:', restored.returncode, flush=True)
    if result['restored']['returncode']:
        raise RuntimeError('Shared server restore failed')


if __name__ == '__main__':
    main()
