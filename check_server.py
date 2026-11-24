"""Inspect the existing environment and verify assets without installing packages."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='0', help='DCU index, or cpu for local checks')
    args = parser.parse_args()
    info = {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'requested_device': args.device,
            'packages': {}, 'assets': []}
    errors = []
    modules = {}
    for name in ('torch', 'torchvision', 'numpy', 'cv2', 'ultralytics',
                 'shapely', 'lap', 'matplotlib', 'yaml'):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            info['packages'][name] = getattr(module, '__version__', 'import OK')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    if 'ultralytics' in modules:
        try:
            solutions = importlib.import_module('ultralytics.solutions')
            for name in ('Heatmap', 'ObjectCounter'):
                if not hasattr(solutions, name):
                    errors.append(f'ultralytics.solutions.{name} is unavailable')
        except Exception as exc:
            errors.append(f'ultralytics.solutions: {type(exc).__name__}: {exc}')
    if 'torch' in modules:
        torch = modules['torch']
        info['accelerator_available'] = torch.cuda.is_available()
        info['hip_version'] = getattr(torch.version, 'hip', None)
        info['device_count'] = torch.cuda.device_count()
        try:
            if args.device == 'cpu':
                device = torch.device('cpu')
                info['device_name'] = 'cpu'
            else:
                if not torch.cuda.is_available():
                    raise RuntimeError('DCU is unavailable in this Python environment')
                index = int(args.device)
                torch.cuda.set_device(index)
                device = torch.device(f'cuda:{index}')
                info['device_name'] = torch.cuda.get_device_name(index)
            x = torch.ones((16, 16), device=device)
            actual = (x @ x)[0, 0].item()
            if actual != 16:
                raise RuntimeError(f'Unexpected matrix multiplication result: {actual}')
            info['tensor_operation'] = 'passed'
        except Exception as exc:
            errors.append(f'device: {type(exc).__name__}: {exc}')
    manifest = json.loads((ROOT / 'assets_manifest.json').read_text(encoding='utf-8'))
    for asset in manifest:
        path = ROOT / asset['path']
        valid = path.is_file() and path.stat().st_size == asset['bytes']
        if valid:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            valid = digest.hexdigest() == asset['sha256']
        info['assets'].append({'path': asset['path'], 'valid': valid})
        if not valid:
            errors.append(f'Missing or corrupted asset: {asset["path"]}')
    if 'cv2' in modules:
        cap = modules['cv2'].VideoCapture(str(ROOT / 'data/general.mp4'))
        try:
            info['video_readable'] = bool(cap.isOpened() and cap.read()[0])
            if not info['video_readable']:
                errors.append('data/general.mp4 cannot be decoded')
        finally:
            cap.release()
    info['errors'] = errors
    info['passed'] = not errors
    directory = ROOT / 'reports'
    directory.mkdir(exist_ok=True)
    (directory / 'environment_check.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if errors:
        print('\nCheck failed. Resolve the listed errors before running the experiments.', file=sys.stderr)
        return 1
    print('\nCheck passed. The complete experiment still needs a smoke run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
