"""Train and validate four distribution-box detector groups with one fixed protocol."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import zipfile

# Keep the shared DCU Python installation unchanged during execution.
os.environ['YOLO_AUTOINSTALL'] = 'false'
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'box_protocol.json'
GROUPS = ('n_base', 'n_tuned', 's_base', 's_tuned')
HYPERPARAMETERS = ('lr0', 'lrf', 'weight_decay', 'warmup_epochs')
TUNED = {'lr0': 0.005, 'lrf': 0.1, 'weight_decay': 0.001, 'warmup_epochs': 1.0}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def inspect_data(path):
    """Count and validate labels, and record file metadata without editing shared data."""
    import yaml
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    names = config.get('names')
    if (not isinstance(names, (list, dict)) or len(names) != 1
            or (isinstance(names, dict) and set(names) != {0})
            or int(config.get('nc', 1)) != 1):
        raise ValueError('Expected one class with ID 0 in data_box.yaml')
    base = Path(config.get('path', path.parent))
    if not base.is_absolute():
        base = path.parent / base
    result = {'config_sha256': digest(path), 'class_names': names, 'splits': {}}
    image_sets = {}
    for split in ('train', 'val'):
        images = Path(config[split])
        if not images.is_absolute():
            images = base / images
        pictures = sorted(p for p in images.rglob('*')
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not pictures:
            raise ValueError('No images in ' + str(images))
        if 'images' not in images.parts:
            raise ValueError('Expected YOLO images/train and images/val directories')
        image_sets[split] = {str(p.resolve()) for p in pictures}
        signature = hashlib.sha256()
        annotated = missing = backgrounds = objects = 0
        for image in pictures:
            parts = list(image.parts)
            index = len(parts) - 1 - parts[::-1].index('images')
            parts[index] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            stat = image.stat()
            # Fingerprint tracks image path/size/mtime plus full label contents.
            signature.update(json.dumps([str(image.resolve()), stat.st_size, stat.st_mtime_ns]).encode())
            if not label.is_file():
                missing += 1
                backgrounds += 1
                signature.update(b'MISSING_LABEL')
                continue
            text = label.read_text(encoding='utf-8-sig')
            signature.update(text.encode('utf-8'))
            lines = [line.split() for line in text.splitlines() if line.strip()]
            if not lines:
                backgrounds += 1
            else:
                annotated += 1
            for fields in lines:
                values = [float(value) for value in fields]
                if (len(values) != 5 or not all(math.isfinite(v) for v in values)
                        or values[0] != 0 or not all(0 <= v <= 1 for v in values[1:])
                        or values[3] <= 0 or values[4] <= 0):
                    raise ValueError('Invalid YOLO detection label: ' + str(label))
                objects += 1
        if annotated == 0:
            raise ValueError('No annotated images in ' + split)
        result['splits'][split] = dict(images=len(pictures), annotated=annotated,
                                      backgrounds=backgrounds, missing_labels=missing,
                                      objects=objects, file_metadata_and_labels_sha256=signature.hexdigest())
    if image_sets['train'] & image_sets['val']:
        raise ValueError('The same image paths occur in train and val')
    return result


def load_protocol():
    if not PROTOCOL.is_file():
        raise FileNotFoundError('First run: python box_experiment.py plan')
    protocol = read_json(PROTOCOL)
    actual = inspect_data(Path(protocol['data']))
    if actual != protocol['dataset']:
        raise ValueError('Dataset configuration, image metadata or labels changed after plan creation')
    return protocol


def show_plan(protocol):
    print('FIXED FOUR-GROUP TRAINING PROTOCOL')
    print('Data:', protocol['data'])
    for split, summary in protocol['dataset']['splits'].items():
        print(split + ':', json.dumps(summary))
    print('Shared training settings:', json.dumps(protocol['common'], sort_keys=True))
    print('\n{:<20} {:>16} {:>16}'.format('Parameter', 'Baseline', 'Tuned'))
    for key in HYPERPARAMETERS:
        print('{:<20} {:>16} {:>16}'.format(key, protocol['baseline'][key], protocol['tuned'][key]))
    print('\nGroups:', ', '.join(GROUPS))
    print('Both sizes use the same optimizer, data, epochs, batch and seed.')
    print('The four changes are tested together, so this is not a one-factor-at-a-time study.')
    print('Validation: same val split; batch=1; imgsz=640; FP32; device=' + str(protocol['common']['device']))


def make_plan(args):
    if PROTOCOL.is_file():
        show_plan(load_protocol())
        print('\nUsing the existing saved protocol; command-line planning defaults do not overwrite it.')
        return
    if args.epochs < 1 or args.batch < 1:
        raise ValueError('epochs and batch must be positive')
    from ultralytics.cfg import DEFAULT_CFG_DICT
    path = ROOT / 'data_box.yaml'
    baseline = {key: float(DEFAULT_CFG_DICT[key]) for key in HYPERPARAMETERS}
    if any(baseline[key] == TUNED[key] for key in HYPERPARAMETERS):
        raise ValueError('Installed defaults differ from this design; a tuned value equals a baseline value')
    weights = {}
    for size in ('n', 's'):
        weight = ROOT / 'models' / ('yolo26' + size + '.pt')
        weights[size] = {'path': str(weight), 'sha256': digest(weight)}
    protocol = {
        'created_utc': now(), 'data': str(path), 'dataset': inspect_data(path),
        'common': dict(epochs=args.epochs, batch=args.batch, imgsz=640, device=args.device,
                       workers=0, amp=False, cache=False, optimizer='SGD',
                       momentum=0.937, seed=0, deterministic=True, patience=0,
                       close_mosaic=0, plots=True, val=True, save=True),
        'baseline': baseline, 'tuned': TUNED, 'pretrained': weights,
        'note': 'Baseline refers to these four default hyperparameters, with shared explicit runtime settings.',
    }
    write_json(PROTOCOL, protocol)
    show_plan(protocol)
    print('\nSaved:', PROTOCOL)


def selected(group):
    return GROUPS if group == 'all' else (group,)


def train(args):
    from ultralytics import YOLO
    import ultralytics
    import torch
    protocol = load_protocol()
    for group in selected(args.group):
        directory = ROOT / 'runs_box' / group
        marker = directory / 'training_complete.json'
        if marker.is_file():
            saved = read_json(marker)
            if saved['protocol_sha256'] != digest(PROTOCOL):
                raise ValueError('Existing run uses a different protocol: ' + group)
            if not (directory / 'weights/best.pt').is_file():
                raise FileNotFoundError('Completed run is missing best.pt: ' + group)
            print('Already completed; skip:', group, flush=True)
            continue
        print('\nSTART TRAINING:', group, now(), flush=True)
        if directory.exists():
            last = directory / 'weights/last.pt'
            if not args.resume or not last.is_file():
                raise ValueError('Partial run exists: {}. To continue its saved last.pt use train --group {} --resume'.format(directory, group))
            saved_plan = read_json(directory / 'requested_experiment.json')
            if saved_plan['protocol_sha256'] != digest(PROTOCOL):
                raise ValueError('Partial run protocol differs from the saved plan')
            model = YOLO(str(last))
            model.train(resume=True, device=protocol['common']['device'])
        else:
            weight = protocol['pretrained'][group[0]]
            if digest(Path(weight['path'])) != weight['sha256']:
                raise ValueError('Pretrained model changed: ' + weight['path'])
            config = dict(protocol['common'])
            config.update(protocol['tuned'] if group.endswith('tuned') else protocol['baseline'])
            config.update(data=protocol['data'], project=str(ROOT / 'runs_box'),
                          name=group, exist_ok=True, resume=False)
            directory.mkdir(parents=True)
            write_json(directory / 'requested_experiment.json',
                       {'group': group, 'protocol_sha256': digest(PROTOCOL), 'parameters': config})
            model = YOLO(weight['path'])
            model.train(**config)
        with (directory / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        final_epoch = int(float({key.strip(): value for key, value in rows[-1].items()}.get('epoch', 0))) if rows else 0
        if final_epoch < protocol['common']['epochs']:
            raise RuntimeError('Training did not reach the planned epoch count: ' + group)
        best = directory / 'weights/best.pt'
        import yaml
        actual_args = yaml.safe_load((directory / 'args.yaml').read_text(encoding='utf-8-sig'))
        expected = dict(protocol['common'])
        expected.update(protocol['tuned'] if group.endswith('tuned') else protocol['baseline'])
        for key, value in expected.items():
            if str(actual_args.get(key)) != str(value) and actual_args.get(key) != value:
                raise RuntimeError('Saved training argument differs for {}: {} != {}'.format(
                    key, actual_args.get(key), value))
        write_json(marker, {'group': group, 'completed_utc': now(),
                            'protocol_sha256': digest(PROTOCOL), 'best_sha256': digest(best),
                            'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
                            'python': platform.python_version()})
        print('TRAINING COMPLETE:', group, 'Best:', best, flush=True)


def validate(args):
    from ultralytics import YOLO
    protocol = load_protocol()
    for group in selected(args.group):
        directory = ROOT / 'runs_box' / group
        completion = read_json(directory / 'training_complete.json')
        if completion['protocol_sha256'] != digest(PROTOCOL):
            raise ValueError('Training protocol mismatch: ' + group)
        best = directory / 'weights/best.pt'
        signature = digest(best)
        report = directory / 'validation.json'
        if report.is_file() and not args.repeat:
            previous = read_json(report)
            if previous['checkpoint_sha256'] == signature and previous['protocol_sha256'] == digest(PROTOCOL):
                print('Already validated; skip:', group, flush=True)
                continue
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        config = dict(data=protocol['data'], split='val', imgsz=640, batch=1,
                      device=protocol['common']['device'], workers=0, half=False,
                      rect=True, conf=0.001, iou=0.7, max_det=300, augment=False,
                      plots=True, save_json=False, project=str(ROOT / 'box_validation'),
                      name=group + '_' + stamp, exist_ok=False)
        print('\nEXPLICIT model.val():', group, flush=True)
        model = YOLO(str(best))
        metrics = model.val(**config)
        values = dict(precision=float(metrics.box.mp), recall=float(metrics.box.mr),
                      mAP50=float(metrics.box.map50), mAP50_95=float(metrics.box.map),
                      inference_ms_per_image=float(metrics.speed['inference']))
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError('Non-finite validation metric: ' + group)
        write_json(report, {'group': group, 'protocol_sha256': digest(PROTOCOL),
                            'checkpoint': str(best), 'checkpoint_sha256': signature,
                            'validated_utc': now(), 'val_settings': config,
                            'metrics': values, 'speed': metrics.speed,
                            'save_dir': str(Path(config['project']) / config['name'])})
        print('VALIDATION COMPLETE:', group, json.dumps(values), flush=True)
    table()


def table():
    print('\n{:<12} {:>9} {:>9} {:>9} {:>12} {:>13}'.format(
        'Group', 'Precision', 'Recall', 'mAP50', 'mAP50-95', 'Inference(ms)'))
    rows = []
    records = []
    for group in GROUPS:
        path = ROOT / 'runs_box' / group / 'validation.json'
        if not path.is_file():
            print('{:<12} NOT VALIDATED'.format(group))
            continue
        record = read_json(path)
        if record['protocol_sha256'] != digest(PROTOCOL):
            raise ValueError('Validation uses a different protocol: ' + group)
        values = record['metrics']
        records.append(record)
        rows.append(dict(group=group, **values))
        print('{:<12} {:>9.4f} {:>9.4f} {:>9.4f} {:>12.4f} {:>13.2f}'.format(
            group, values['precision'], values['recall'], values['mAP50'],
            values['mAP50_95'], values['inference_ms_per_image']))
    if records:
        comparable = [{key: value for key, value in item['val_settings'].items()
                       if key not in ('project', 'name')} for item in records]
        if any(settings != comparable[0] for settings in comparable[1:]):
            raise ValueError('Saved validation settings differ between groups')
    if rows:
        target = ROOT / 'reports/box_comparison.csv'
        target.parent.mkdir(exist_ok=True)
        with target.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('Saved:', target)
    print('Values 0..1 are fractions; inference time excludes preprocess and postprocess.')


def pack():
    load_protocol()
    table()
    files = {}

    def add(source, relative):
        if not source.is_file():
            raise FileNotFoundError(str(source))
        files['box_report_materials/' + str(relative).replace('\\', '/')] = source

    for name in ('box_experiment.py', 'box_protocol.json', 'data_box.yaml',
                 'BOX_GUIDE.md', 'reports/box_comparison.csv'):
        add(ROOT / name, name)
    for group in GROUPS:
        directory = ROOT / 'runs_box' / group
        record = read_json(directory / 'validation.json')
        for source in directory.iterdir():
            if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES | {'.json', '.yaml', '.csv'}:
                add(source, Path('runs_box') / group / source.name)
        add(directory / 'weights/best.pt', Path('runs_box') / group / 'weights/best.pt')
        validation = Path(record['save_dir'])
        for source in validation.iterdir():
            if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES | {'.json', '.yaml', '.csv'}:
                add(source, Path('validation') / group / source.name)
        log = ROOT / 'logs' / (group + '.log')
        if log.is_file():
            add(log, Path('logs') / log.name)
    earlier = ROOT / 'report_materials'
    if earlier.is_dir():
        for source in earlier.rglob('*'):
            if source.is_file():
                add(source, Path('earlier_results') / source.relative_to(earlier))
    archive_path = ROOT / 'box_report_materials.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
    print('Report materials, source code and best checkpoints:', archive_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan', help='Save and display the common training protocol')
    plan.add_argument('--epochs', type=int, default=10)
    plan.add_argument('--batch', type=int, default=16)
    plan.add_argument('--device', default='0')
    training = sub.add_parser('train')
    training.add_argument('--group', choices=GROUPS + ('all',), default='all')
    training.add_argument('--resume', action='store_true')
    validation = sub.add_parser('validate')
    validation.add_argument('--group', choices=GROUPS + ('all',), default='all')
    validation.add_argument('--repeat', action='store_true')
    sub.add_parser('table')
    sub.add_parser('pack')
    args = parser.parse_args()
    try:
        if args.action == 'plan':
            make_plan(args)
        elif args.action == 'train':
            train(args)
        elif args.action == 'validate':
            validate(args)
        elif args.action == 'table':
            table()
        else:
            pack()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, '\nOperation stopped: {}\n'.format(exc))


if __name__ == '__main__':
    main()
