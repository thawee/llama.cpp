#!/usr/bin/env python3
"""Install the private local runtime without starting a service."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import sys
import tempfile
import time

MARKER = 'llama-local managed v1'
ALIAS = 'qwen3.8-27b-local'
LABEL = 'local.llama.qwen27b'


def read_json(path):
    value = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def pi_config(value):
    value = copy.deepcopy(value)
    providers = value.setdefault('providers', {})
    provider = providers.setdefault('llama', {})
    models = provider.get('models', [])
    if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
        raise ValueError('Pi llama models must be an array of objects')
    model = {'id': ALIAS, 'name': 'Local Qwen 27B', 'contextWindow': 131072,
             'maxTokens': 8192, 'reasoning': True, 'input': ['text'],
             'cost': dict.fromkeys(['input', 'output', 'cacheRead', 'cacheWrite'], 0)}
    provider.update(baseUrl='http://127.0.0.1:8000/v1', api='openai-completions', apiKey='local')
    provider['models'] = [m for m in models if m.get('id') != ALIAS] + [model]
    return value


def atomic_write(path, data, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    data = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists() and path.read_bytes() == data:
        return
    if path.exists():
        backup = path.with_suffix(path.suffix + '.bak')
        if backup.exists():
            backup = path.with_name(path.name + '.bak.' + str(time.time_ns()))
        shutil.copy2(path, backup)
    atomic_write(path, data)


def check_owned(path):
    if path.is_symlink() or (path.exists() and MARKER not in path.read_text()):
        raise RuntimeError(f'Refusing to replace unrelated command: {path}')


def install(args):
    home = Path.home()
    prefix = args.prefix.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    model = args.model.expanduser().resolve(strict=True)
    if not model.is_file():
        raise ValueError(f'Model is not a file: {model}')
    executable = runtime / 'llama-server'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f'Executable missing: {executable}')
    files = [executable] + sorted(runtime.glob('*.dylib'))
    if not files[1:]:
        raise ValueError('Runtime must include its dylibs')
    root = prefix / 'lib/llama-local'
    config = home / '.config/llama-local/config.json'
    pi = home / '.pi/agent/models.json'
    settings = read_json(config)
    mode = getattr(args, 'speculative', None) or settings.get('speculative', 'ngram-simple')
    if mode not in ('ngram-simple', 'draft-mtp'):
        raise ValueError('speculative must be ngram-simple or draft-mtp')
    merged_pi = pi_config(read_json(pi))
    controller_source = Path(__file__).with_name('llama-local.py').read_bytes()
    commands = ['llama-server', 'llama-local', 'llama-pi', 'llama-copilot']
    for name in commands:
        check_owned(prefix / 'bin' / name)
    controller = root / 'controller.py'
    check_owned(controller)
    plist = home / 'Library/LaunchAgents' / (LABEL + '.plist')
    if plist.exists() and plistlib.loads(plist.read_bytes()).get('LlamaLocalOwner') != MARKER:
        raise RuntimeError(f'Refusing unrelated LaunchAgent: {plist}')
    current = root / 'current'
    if current.exists() and not current.is_symlink():
        raise RuntimeError(f'Refusing unrelated runtime directory: {current}')
    digest = hashlib.sha256()
    for source in files:
        digest.update(source.name.encode())
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    release = root / 'releases' / digest.hexdigest()[:16]
    if not release.exists():
        release.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.stage-', dir=release.parent))
        for source in files:
            shutil.copy2(source, staging / source.name)
        staging.rename(release)
    link = root / ('.current-' + str(os.getpid()))
    link.symlink_to(release)
    os.replace(link, current)
    atomic_write(controller, controller_source, 0o700)
    settings.update(runtime=str(current), model=str(model), speculative=mode, alias=ALIAS, plist=str(plist),
                    controller=str(controller), python=str(Path(sys.executable).resolve()),
                    state=str(home / '.local/state/llama-local'))
    write_json(config, settings)
    write_json(pi, merged_pi)
    state = Path(settings['state'])
    state.mkdir(parents=True, exist_ok=True)
    agent = {'Label': LABEL, 'LlamaLocalOwner': MARKER, 'RunAtLoad': False, 'KeepAlive': False,
             'ProgramArguments': [str(Path(sys.executable).resolve()), str(controller), '--config', str(config), 'run'],
             'StandardOutPath': str(state / 'server.log'), 'StandardErrorPath': str(state / 'server.log')}
    atomic_write(plist, plistlib.dumps(agent))
    for name, command in zip(commands, ['raw', None, 'pi', 'copilot']):
        argv = [str(Path(sys.executable).resolve()), str(controller), '--config', str(config)]
        if command:
            argv.append(command)
        wrapper = '#!/bin/sh\n# ' + MARKER + '\nexec ' + shlex.join(argv) + ' "$@"\n'
        atomic_write(prefix / 'bin' / name, wrapper.encode(), 0o755)
    print(f'Installed {release}; no service started. Commands: {prefix / "bin"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path.home() / '.local')
    parser.add_argument('--runtime', type=Path, required=True, help='Verified bin directory with executable and dylibs')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--speculative', choices=['ngram-simple', 'draft-mtp'], help='Preserve the installed profile when omitted')
    try:
        install(parser.parse_args())
    except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as error:
        sys.exit(str(error))
