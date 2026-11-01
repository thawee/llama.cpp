#!/usr/bin/env python3
"""llama-local managed v1: on-demand private launchd runtime."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

LABEL = 'local.llama.qwen27b'
ALIAS = 'qwen3.8-27b-local'
BASE = 'http://127.0.0.1:8000'
DOMAIN = f'gui/{os.getuid()}'
JOB = DOMAIN + '/' + LABEL


def launch(*args):
    return subprocess.run(['/bin/launchctl', *args], capture_output=True, text=True, timeout=15)


def job_pid(config):
    result = launch('print', JOB)
    if result.returncode == 0:
        for expected in [config['plist'], config['controller'], config['python']]:
            if expected not in result.stdout:
                raise RuntimeError('Loaded LaunchAgent identity differs from installed runtime')
    match = re.search(r'^\s*pid = (\d+)', result.stdout, re.M)
    return int(match[1]) if match else None


def listeners():
    result = subprocess.run(['/usr/sbin/lsof', '-nP', '-t', '-iTCP:8000', '-sTCP:LISTEN'],
                            capture_output=True, text=True, timeout=10)
    if result.returncode not in (0, 1):
        raise RuntimeError('Cannot establish port ownership: ' + result.stderr.strip())
    return {int(pid) for pid in result.stdout.split()}


def ready():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def get(path):
            with opener.open(BASE + path, timeout=2) as response:
                return json.load(response)
        if get('/health').get('status') != 'ok':
            return False
        models = get('/v1/models')['data']
        props = get('/props')
        return (any(m['id'] == ALIAS for m in models)
                and props['default_generation_settings']['n_ctx'] == 131072)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def start(config):
    occupied = listeners()
    pid = job_pid(config)
    if occupied and occupied != {pid}:
        raise RuntimeError('Port 8000 belongs to another process; it was left untouched')
    if not pid:
        if launch('print', JOB).returncode:
            result = launch('bootstrap', DOMAIN, config['plist'])
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
        result = launch('kickstart', JOB)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        pid = job_pid(config)
        occupied = listeners()
        if occupied and occupied != {pid}:
            raise RuntimeError('Port ownership changed; foreign process left untouched')
        if pid and occupied == {pid} and ready():
            print(f'Ready: {ALIAS}, 131072 tokens, {BASE}', file=sys.stderr, flush=True)
            return
        time.sleep(1)
    raise RuntimeError('Server not ready after 120 seconds; inspect llama-local logs')


def stop(config):
    pid = job_pid(config)
    if launch('print', JOB).returncode:
        print('Stopped')
        return
    result = launch('bootout', JOB)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    deadline = time.monotonic() + 15
    while pid and time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        if pid:
            raise RuntimeError('LaunchAgent unloaded but process has not exited')
    print('Stopped')


def client_env(client):
    env = dict(os.environ)
    if client == 'pi':
        env['PI_OFFLINE'] = '1'
        return env
    registry = env.get('COPILOT_PROVIDERS_CONFIG')
    paths = [Path(os.environ.get('COPILOT_HOME', str(Path.home() / '.copilot'))) / 'providers.json']
    if registry:
        paths.append(Path(registry))
    for path in paths:
        if path.exists():
            value = json.loads(path.read_text())
            if value:
                raise RuntimeError(f'Conflicting Copilot provider registry: {path}')
        elif registry and path == Path(registry):
            raise RuntimeError(f'Cannot read Copilot provider registry: {path}')
    for key in list(env):
        if key.startswith(('COPILOT_PROVIDER_', 'COPILOT_PROVIDERS_', 'OPENAI_', 'AZURE_OPENAI_', 'ANTHROPIC_')):
            del env[key]
    env.update(COPILOT_PROVIDER_BASE_URL=BASE + '/v1', COPILOT_PROVIDER_TYPE='openai',
               COPILOT_PROVIDER_WIRE_API='completions', COPILOT_PROVIDER_TRANSPORT='http',
               COPILOT_MODEL=ALIAS, COPILOT_PROVIDER_WIRE_MODEL=ALIAS,
               COPILOT_PROVIDER_MAX_PROMPT_TOKENS='110592', COPILOT_PROVIDER_MAX_OUTPUT_TOKENS='8192',
               COPILOT_OFFLINE='true')
    return env


def run(config, extra):
    speculative = os.environ.get('LLAMA_LOCAL_SPECULATIVE', '1')
    if speculative not in ('0', '1'):
        raise ValueError('LLAMA_LOCAL_SPECULATIVE must be 0 or 1')
    mode = config.get('speculative', 'ngram-simple')
    if mode not in ('ngram-simple', 'draft-mtp'):
        raise ValueError('speculative must be ngram-simple or draft-mtp')
    args = ['--model', os.environ.get('LLAMA_LOCAL_MODEL', config['model']), '--alias', ALIAS, '--offline', '--host', '127.0.0.1',
            '--port', '8000', '-np', '1', '-c', '131072', '-ngl', '99', '-t', '5', '-tb', '5',
            '-b', '64', '-ub', '64', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '--reasoning', 'on',
            '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0',
            '--presence-penalty', '0', '--repeat-penalty', '1.0']
    if mode == 'draft-mtp':
        args += ['--fit', 'off']
        if speculative == '1':
            args += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '5',
                     '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0']
    else:
        args += ['--memory-budget', '18432']
    if speculative == '1' and mode == 'ngram-simple':
        args += ['--spec-type', 'ngram-simple', '--spec-ngram-simple-size-n', '3', '--spec-ngram-simple-size-m', '3']
    execute(config, args + extra)


def execute(config, args):
    binary = str(Path(config['runtime']) / 'llama-server')
    env = dict(os.environ, DYLD_LIBRARY_PATH=config['runtime'])
    os.execve(binary, [binary, *args], env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path.home() / '.config/llama-local/config.json')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'logs', 'run', 'raw', 'pi', 'copilot'])
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra
    config = json.loads(args.config.read_text())
    command = args.command
    if command in ('pi', 'copilot') and any(arg in ('--help', '-h', '--version') for arg in extra):
        os.execvpe(command, [command, *extra], client_env(command))
    if command == 'raw':
        execute(config, extra)
    if command == 'run':
        run(config, extra)
    if command in ('pi', 'copilot'):
        env = client_env(command)
    state = Path(config['state'])
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if command in ('start', 'pi', 'copilot'):
            start(config)
        elif command == 'stop':
            stop(config)
        elif command == 'status':
            pid = job_pid(config)
            owned = bool(pid and listeners() == {pid})
            print(f'pid={pid or "none"} owned={owned} ready={owned and ready()}')
        elif command == 'logs':
            subprocess.run(['/usr/bin/tail', '-n', '50', str(state / 'server.log')], check=True)
    if command in ('pi', 'copilot'):
        argv = (['pi', '--provider', 'llama', '--model', ALIAS, '--thinking', 'off']
                if command == 'pi' else ['copilot'])
        os.execvpe(command, argv + extra, env)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
