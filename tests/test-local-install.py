import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load('installer', 'install-local-runtime.py')
controller = load('controller', 'llama-local.py')


class InstallTests(unittest.TestCase):
    def test_mtp_profile_uses_explicit_context_and_quantized_draft(self):
        for switch in ('0', '1'):
            with patch.dict(os.environ, {'LLAMA_LOCAL_SPECULATIVE': switch}, clear=True), \
                 patch.object(controller, 'execute') as execute:
                controller.run({'model': '/q8head.gguf', 'speculative': 'draft-mtp'}, [])
                flags = execute.call_args.args[1]
                self.assertNotIn('--memory-budget', flags)
                self.assertEqual(flags[flags.index('--fit') + 1], 'off')
                self.assertEqual(flags[flags.index('-c') + 1], '131072')
                self.assertEqual('--spec-type' in flags, switch == '1')
                if switch == '1':
                    self.assertEqual(flags[flags.index('--spec-type') + 1], 'draft-mtp')
                    self.assertEqual(flags[flags.index('--spec-draft-n-max') + 1], '5')
                    self.assertEqual(flags[flags.index('--spec-draft-type-k') + 1], 'q8_0')
                    self.assertEqual(flags[flags.index('--spec-draft-type-v') + 1], 'q8_0')

    def test_merge_preserves_models_and_settings(self):
        original = {'providers': {'other': {'custom': True}, 'llama': {
            'extra': 7, 'models': [{'id': 'custom'}, {'id': installer.ALIAS, 'name': 'old'}]}}}
        merged = installer.pi_config(original)
        self.assertEqual(merged['providers']['other'], {'custom': True})
        self.assertEqual(merged['providers']['llama']['extra'], 7)
        self.assertEqual(merged['providers']['llama']['models'][0], {'id': 'custom'})
        self.assertEqual(merged['providers']['llama']['models'][1]['contextWindow'], 131072)

    def test_atomic_backup_and_idempotence(self):
        with tempfile.TemporaryDirectory(prefix='llama space ') as temp:
            path = Path(temp) / 'models.json'
            path.write_text('{"old": true}')
            installer.write_json(path, {'new': True})
            self.assertEqual(json.loads(path.with_suffix('.json.bak').read_text()), {'old': True})
            installer.write_json(path, {'new': True})
            self.assertEqual(json.loads(path.with_suffix('.json.bak').read_text()), {'old': True})

    def test_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'llama-local'
            path.write_text('# unrelated command')
            with self.assertRaises(RuntimeError):
                installer.check_owned(path)

    def test_client_environment(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
            'HOME': temp, 'COPILOT_PROVIDER_API_KEY': 'secret',
            'COPILOT_PROVIDER_HEADERS': 'secret', 'COPILOT_PROVIDER_MODEL_ID': 'cloud',
            'OPENAI_API_KEY': 'secret'}, clear=True):
            env = controller.client_env('copilot')
            self.assertNotIn('COPILOT_PROVIDER_API_KEY', env)
            self.assertNotIn('COPILOT_PROVIDER_HEADERS', env)
            self.assertNotIn('COPILOT_PROVIDER_MODEL_ID', env)
            self.assertNotIn('OPENAI_API_KEY', env)
            self.assertEqual(env['COPILOT_PROVIDER_BASE_URL'], 'http://127.0.0.1:8000/v1')
            self.assertEqual(env['COPILOT_OFFLINE'], 'true')

    def test_copilot_registry_respects_home(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'COPILOT_HOME': temp}, clear=True):
            (Path(temp) / 'providers.json').write_text('{"providers": ["cloud"]}')
            with self.assertRaises(RuntimeError):
                controller.client_env('copilot')

    def test_foreign_port_never_starts_job(self):
        with patch.object(controller, 'listeners', return_value={99}), \
             patch.object(controller, 'job_pid', return_value=None), \
             patch.object(controller, 'launch') as launch:
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                controller.start({})
            launch.assert_not_called()

    def test_ready_owned_job_is_reused(self):
        with patch.object(controller, 'listeners', return_value={99}), \
             patch.object(controller, 'job_pid', return_value=99), \
             patch.object(controller, 'ready', return_value=True), \
             patch.object(controller, 'launch') as launch:
            controller.start({})
            launch.assert_not_called()

    def test_start_bootstraps_and_kickstarts_once(self):
        missing = subprocess.CompletedProcess([], 113, '', 'not found')
        success = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(controller, 'listeners', side_effect=[set(), {99}]), \
             patch.object(controller, 'job_pid', side_effect=[None, 99]), \
             patch.object(controller, 'ready', return_value=True), \
             patch.object(controller, 'launch', side_effect=[missing, success, success]) as launch:
            controller.start({'plist': '/owned.plist'})
            self.assertEqual([call.args for call in launch.call_args_list], [
                ('print', controller.JOB), ('bootstrap', controller.DOMAIN, '/owned.plist'),
                ('kickstart', controller.JOB)])

    def test_loaded_foreign_job_identity_is_rejected(self):
        output = subprocess.CompletedProcess([], 0, 'path = /foreign.plist\npid = 99', '')
        with patch.object(controller, 'launch', return_value=output):
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                controller.job_pid({'plist': '/owned.plist', 'controller': '/owned.py', 'python': '/python'})

    def test_stop_only_unloads_owned_job(self):
        success = subprocess.CompletedProcess([], 0, '', '')
        with patch.object(controller, 'job_pid', return_value=99), \
             patch.object(controller, 'launch', return_value=success) as launch, \
             patch.object(controller.os, 'kill', side_effect=ProcessLookupError):
            controller.stop({})
            self.assertEqual([call.args for call in launch.call_args_list], [
                ('print', controller.JOB), ('bootout', controller.JOB)])

    def test_speculation_switch_and_flags(self):
        for switch in ('0', '1'):
            with patch.dict(os.environ, {'LLAMA_LOCAL_SPECULATIVE': switch}), \
                 patch.object(controller, 'execute') as execute:
                controller.run({'model': '/model.gguf'}, [])
                flags = execute.call_args.args[1]
                self.assertEqual('--spec-type' in flags, switch == '1')
                self.assertIn('131072', flags)
                if switch == '1':
                    self.assertIn('--spec-ngram-simple-size-n', flags)

    def test_install_temp_home_and_idempotent_wrappers(self):
        with tempfile.TemporaryDirectory(prefix='llama spaces ') as temp:
            home = Path(temp)
            runtime = home / 'build bin'
            runtime.mkdir()
            binary = runtime / 'llama-server'
            binary.write_text('#!/bin/sh\nprintf "raw=%s\\n" "$1"\n')
            binary.chmod(0o755)
            (runtime / 'libtest.dylib').write_text('test library')
            model = home / 'model file.gguf'
            model.write_text('model')
            prefix = home / 'local prefix'
            script = Path(installer.__file__)
            args = [sys.executable, str(script), '--prefix', str(prefix), '--runtime', str(runtime), '--model', str(model)]
            env = dict(os.environ, HOME=str(home))
            for _ in range(2):
                result = subprocess.run(args, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            wrapper = prefix / 'bin/llama-server'
            result = subprocess.run([str(wrapper), '--help'], env=env, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), 'raw=--help')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((home / '.pi/agent/models.json.bak').exists())
            installed = json.loads((home / '.config/llama-local/config.json').read_text())
            self.assertEqual(installed['model'], str(model.resolve()))
            self.assertTrue((prefix / 'lib/llama-local/current').is_symlink())
            pi = prefix / 'bin/pi'
            pi.write_text('#!/bin/sh\nprintf "client=%s\\n" "$1"\n')
            pi.chmod(0o755)
            env['PATH'] = str(prefix / 'bin') + os.pathsep + env.get('PATH', '')
            result = subprocess.run([str(prefix / 'bin/llama-pi'), '--help'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'client=--help')
            self.assertFalse((home / '.local/state/llama-local/control.lock').exists())

    def test_invalid_config_prevents_mutation(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'HOME': temp}):
            home = Path(temp)
            pi = home / '.pi/agent/models.json'
            pi.parent.mkdir(parents=True)
            pi.write_text('{invalid')
            runtime = home / 'runtime'
            runtime.mkdir()
            binary = runtime / 'llama-server'
            binary.write_text('#!/bin/sh\n')
            binary.chmod(0o755)
            (runtime / 'libtest.dylib').touch()
            model = home / 'model.gguf'
            model.touch()
            with self.assertRaises(ValueError):
                installer.install(type('Args', (), {'prefix': home / 'prefix', 'runtime': runtime, 'model': model})())
            self.assertFalse((home / 'prefix').exists())


if __name__ == '__main__':
    unittest.main()
