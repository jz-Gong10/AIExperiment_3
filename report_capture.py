"""Display saved evidence and collect existing output images; never train a model."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def heading(title):
    print('\n' + '=' * 64)
    print(title)
    print('=' * 64)


def environment():
    path = ROOT / 'reports/environment_check.json'
    info = read_json(path)
    heading('SAVED SERVER ENVIRONMENT CHECK')
    print('Source:', path)
    print('Python:', info['python'].splitlines()[0])
    print('Executable:', info['executable'])
    print('Platform:', info['platform'])
    for name, version in info['packages'].items():
        print(name + ':', version)
    for key in ('requested_device', 'accelerator_available', 'hip_version',
                'device_count', 'device_name', 'tensor_operation',
                'video_readable', 'errors', 'passed'):
        print(key + ':', info.get(key))


def dataset():
    import yaml
    path = ROOT / 'data_box.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    heading('DATASET CONFIGURATION AND FILE COUNTS')
    print('Config:', path)
    print(path.read_text(encoding='utf-8-sig').strip())
    base = Path(config.get('path', path.parent))
    if not base.is_absolute():
        base = path.parent / base
    for split in ('train', 'val', 'test'):
        relative = config.get(split)
        if not isinstance(relative, str):
            continue
        images = Path(relative)
        if not images.is_absolute():
            images = base / images
        if not images.is_dir():
            print(split + ': cannot count image-directory entries at', images)
            continue
        labels = base / 'labels' / images.name
        pictures = sorted(p for p in images.rglob('*')
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        label_files = list(labels.rglob('*.txt')) if labels.is_dir() else []
        print('{}: {} image files, {} label files'.format(
            split, len(pictures), len(label_files)))
    print('Counts above are files, not object instances; see training scan for background images.')


def select_app(app):
    paths = sorted((ROOT / 'results' / app / 'baseline').glob('*/summary.json'))
    if not paths:
        raise FileNotFoundError('No saved baseline run for ' + app)
    runs = [(path, read_json(path)) for path in paths]
    complete = [entry for entry in runs if entry[1].get('completed_video')]
    return (complete or runs)[-1]


def applications():
    heading('SAVED BASELINE APPLICATION RESULTS')
    print('Selection: most recent complete run, or latest partial run if none completed.')
    for app in ('heatmap', 'counting'):
        path, info = select_app(app)
        print('\nApplication:', app)
        print('Run:', info['run_id'])
        print('Device:', info['parameters'].get('device'))
        print('Model:', Path(info['parameters']['model']).name)
        print('Processed frames:', info['frames_processed'])
        print('Completed video:', info['completed_video'])
        for key in ('mean_process_ms', 'processing_fps'):
            value = info.get(key)
            print(key + ':', '{:.2f}'.format(value) if value is not None else 'not measured')
        metrics = info.get('last_metrics', {})
        print('Final IN / OUT:', metrics.get('in_count'), '/', metrics.get('out_count'))
        print('Output:', path.parent / 'output.mp4')


def training(run_name):
    import yaml
    directory = ROOT / 'runs_box' / run_name
    heading('SAVED TRAINING RUN: ' + run_name)
    print('Directory:', directory)
    config = yaml.safe_load((directory / 'args.yaml').read_text(encoding='utf-8-sig'))
    for key in ('model', 'data', 'epochs', 'imgsz', 'batch', 'device',
                'workers', 'amp', 'optimizer', 'lr0', 'lrf', 'momentum',
                'weight_decay', 'warmup_epochs', 'seed'):
        print(key + ':', config.get(key))
    with (directory / 'results.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if rows:
        print('\nLAST SAVED EPOCH (not necessarily the best checkpoint):')
        for key, value in rows[-1].items():
            if key == 'epoch' or key.startswith('metrics/'):
                print(key + ':', value)
    print('\nBest checkpoint exists:', (directory / 'weights/best.pt').is_file())
    print('Saved optimizer setting is a requested option; auto may choose a different optimizer.')
    print('Inference speed is not in results.csv; retain the validation terminal screenshot.')


def figures(run_name):
    import cv2
    directory = ROOT / 'report_materials'
    directory.mkdir(exist_ok=True)
    selected = [(app,) + select_app(app) for app in ('heatmap', 'counting')]
    frame_number = min([1000] + [int(info['frames_processed']) for _, _, info in selected])
    if frame_number < 1:
        raise RuntimeError('Saved runs have no processed frames')
    for app, summary, info in selected:
        cap = cv2.VideoCapture(str(summary.parent / 'output.mp4'))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Cannot decode frame from ' + str(summary.parent / 'output.mp4'))
            target = directory / '{}_frame_{:06d}.jpg'.format(app, frame_number)
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError('Cannot save ' + str(target))
            print('Real output frame:', target)
        finally:
            cap.release()
        for name in ('summary.json', 'metrics.csv'):
            source = summary.parent / name
            if source.is_file():
                shutil.copy2(source, directory / (app + '_' + name))
    training_dir = ROOT / 'runs_box' / run_name
    if training_dir.is_dir():
        target_dir = directory / run_name
        target_dir.mkdir(exist_ok=True)
        for source in training_dir.iterdir():
            if source.is_file() and (source.suffix.lower() in IMAGE_SUFFIXES
                                    or source.name in ('args.yaml', 'results.csv')):
                shutil.copy2(source, target_dir / source.name)
                print('Copied actual training output:', target_dir / source.name)
    for source in (ROOT / 'data_box.yaml', ROOT / 'reports/environment_check.json'):
        if source.is_file():
            shutil.copy2(source, directory / source.name)
    archive_path = ROOT / 'report_materials.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(directory.rglob('*')):
            if source.is_file():
                archive.write(source, source.relative_to(ROOT))
    print('\nMaterials archive:', archive_path)
    print('Download these files to view the images. This command does not open a display.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('section', choices=('env', 'data', 'apps', 'train', 'figures'))
    parser.add_argument('--run', default='smoke_n', help='Actual run folder under runs_box')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('--run must be a single directory name')
    try:
        actions = {'env': environment, 'data': dataset, 'apps': applications,
                   'train': lambda: training(args.run), 'figures': lambda: figures(args.run)}
        actions[args.section]()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, 'Cannot collect this section: {}\n'.format(exc))


if __name__ == '__main__':
    main()
