"""Official solution runner. Every run saves its parameters, metrics and video."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from app_specs import APPS, VARIANTS, MODEL_PAIRS

ROOT = Path(__file__).resolve().parent

def read_settings():
    return json.loads((ROOT / 'settings.json').read_text(encoding='utf-8-sig'))

def source_path(app, settings):
    p = Path(settings['sources'][app]).expanduser()
    return p if p.is_absolute() else ROOT / p

def parameters(app, variant, settings, width, height, fps):
    """Translate normalized region coordinates into the original video's pixels."""
    spec = APPS[app]
    model = MODEL_PAIRS[spec['task']][variant == 'model_compare']
    cfg = dict(model=str(ROOT / 'models' / model), conf=0.25, imgsz=640,
               classes=None, device=settings.get('device', 'cpu'), show=False,
               show_conf=True, line_width=2, verbose=False, tracker='bytetrack.yaml')
    cfg.update(spec.get('baseline', {}))
    extra = dict(spec.get(variant, {})) if variant in ('change1', 'change2') else {}
    region_variant = extra.pop('region_variant', 'baseline')
    scale_multiplier = extra.pop('scale_multiplier', 1.0)
    cfg.update(extra)
    if app == 'counting':
        ratio = 0.65 if region_variant == 'shifted' else 0.5
        cfg['region'] = [(int(width * ratio), int(height * 0.1)),
                         (int(width * ratio), int(height * 0.9))]
    elif app == 'region_counting':
        def rectangle(a, b, c, d):
            return [(int(width * x), int(height * y)) for x, y in [(a,b),(c,b),(c,d),(a,d)]]
        cfg['region'] = ({'Left': rectangle(.1,.15,.49,.85), 'Right': rectangle(.51,.15,.9,.85)}
                         if region_variant == 'split' else {'Center': rectangle(.2,.15,.8,.85)})
    elif app == 'parking':
        p = ROOT / 'data' / 'parking_regions.json'
        if not p.is_file():
            raise FileNotFoundError('请先执行 python annotate_scene.py parking 标注真实车位。')
        regions = json.loads(p.read_text(encoding='utf-8-sig'))
        if not regions or not all(len(r.get('points', [])) >= 3 for r in regions):
            raise ValueError('车位 JSON 为空或格式错误，请重新标注。')
        for r in regions:
            if any(not (0 <= x < width and 0 <= y < height) for x,y in r['points']):
                raise ValueError('车位坐标超出当前视频范围，请使用当前素材重新标注。')
        cfg['json_file'] = str(p)
    elif app == 'speed':
        scale = settings.get('speed_meter_per_pixel')
        calibrated = scale is not None
        if calibrated:
            if not math.isfinite(float(scale)) or float(scale) <= 0:
                raise ValueError('speed_meter_per_pixel 必须为正数。')
            if not settings.get('speed_calibration_note'):
                raise ValueError('请填写 speed_calibration_note，记录标定依据。')
        else:
            scale = settings.get('speed_demo_meter_per_pixel', 0.05)
        cfg.update(fps=fps, max_speed=300, meter_per_pixel=float(scale) * scale_multiplier)
        cfg['_measurement_status'] = ('calibrated_approximation' if calibrated and scale_multiplier == 1
                                      else 'scale_sensitivity_only' if scale_multiplier != 1
                                      else 'uncalibrated_demo')
    elif app == 'workout':
        cfg['kpts'] = settings.get('workout_keypoints', [5, 7, 9])
    return cfg

def jsonable(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)

def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def choose_app():
    for key,spec in APPS.items():
        print(f"{spec['number']}. {spec['name']} ({key})")
    chosen = input('输入应用编号：').strip()
    return next((k for k,v in APPS.items() if str(v['number']) == chosen), chosen)

def choose_variant():
    for i,(key,title) in enumerate(VARIANTS.items(),1):
        print(f'{i}. {title} ({key})')
    choice = input('输入对比组编号 [1]：').strip() or '1'
    return dict(zip(map(str,range(1,5)),VARIANTS)).get(choice, choice)

def main(default_app=None):
    parser=argparse.ArgumentParser(description='DCU 热力图与目标计数实验')
    parser.add_argument('--app', choices=APPS, default=default_app)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--device', default=None, help='设备编号，例如 0；本地检查可用 cpu')
    parser.add_argument('--no-display', action='store_true', help='仅用于检查或自动记录，不支持距离目标选择')
    args=parser.parse_args()
    app=args.app or choose_app()
    if app not in APPS:
        parser.error('应用编号或名称无效。')
    variant=args.variant or choose_variant()
    if variant not in VARIANTS:
        parser.error('对比组无效。')
    settings=read_settings()
    if args.device is not None:
        settings['device']=args.device
    source=source_path(app, settings)
    if not source.is_file():
        raise FileNotFoundError(f'请准备输入视频并检查 settings.json：{source}')
    import cv2
    import torch
    import ultralytics
    from ultralytics import solutions
    cap=cv2.VideoCapture(str(source))
    writer=None
    try:
        if not cap.isOpened():
            raise RuntimeError(f'无法读取视频：{source}')
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('无法读取有效分辨率或帧率，请先将素材转为常规 MP4 视频。')
        cfg=parameters(app,variant,settings,width,height,fps)
        status=cfg.pop('_measurement_status',None)
        model=Path(cfg['model'])
        if not model.is_file():
            raise FileNotFoundError(f'缺少模型 {model.name}，请核对 models 目录或重新下载完整仓库。')
        print(f"\n应用：{APPS[app]['name']}  对比组：{VARIANTS[variant]}")
        print(json.dumps(cfg,ensure_ascii=False,indent=2))
        if status and status != 'calibrated_approximation':
            print('注意：当前速度仅用于未标定演示或尺度敏感性分析，不能写成真实车辆速度。')
        if args.no_display:
            print('无窗口模式：处理视频并保存结果，不进行交互截图。')
        else:
            print('空格暂停/继续，S 截图，Q/Esc 退出。')
        run_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out=ROOT/'results'/app/variant/run_id;out.mkdir(parents=True)
        shots=ROOT/'screenshots'/app/variant/run_id;shots.mkdir(parents=True)
        calculator=getattr(solutions,APPS[app]['class'])(**cfg)
        writer=cv2.VideoWriter(str(out/'output.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():
            raise RuntimeError('结果视频创建失败，请检查路径和 MP4 编码支持。')
        window='Ultralytics Lab'
        if not args.no_display:
            cv2.namedWindow(window,cv2.WINDOW_NORMAL)
            # Keep the original image aspect ratio; OpenCV maps mouse positions to image coordinates.
            scale=min(1.0,1280/width,800/height)
            cv2.resizeWindow(window,int(width*scale),int(height*scale))
            if app=='distance':
                cv2.setMouseCallback(window,calculator.mouse_event_for_distance)
        timings=[];frames=0;last_metrics={};stop=False;completed=False;snapshots=0
        fieldnames=['frame','video_time_s','process_ms','total_tracks','metrics_json']
        max_frames=args.max_frames if args.max_frames is not None else settings.get('max_frames')
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            rows=csv.DictWriter(f,fieldnames=fieldnames);rows.writeheader()
            while not stop:
                ok,im0=cap.read()
                if not ok:
                    completed=True;break
                t=time.perf_counter();result=calculator(im0);elapsed=time.perf_counter()-t
                frames+=1
                if frames>10:timings.append(elapsed)
                keys={'distance':['pixels_distance'],'heatmap':['in_count','out_count','classwise_count'],
                      'counting':['in_count','out_count','classwise_count'],
                      'parking':['filled_slots','available_slots'],'speed':['speed_dict'],
                      'workout':['workout_count','workout_stage','workout_angle'],
                      'segmentation':[],'region_counting':['region_counts']}[app]
                metrics={k:getattr(result,k,None) for k in keys}
                metrics['track_ids']=list(calculator.track_ids)
                if app=='distance':
                    selected=list(calculator.selected_boxes)
                    valid=len(selected)==2 and all(tid in calculator.track_ids for tid in selected)
                    metrics['selected_ids']=selected;metrics['distance_valid']=valid
                    if not valid:metrics['pixels_distance']=None
                if app=='segmentation':
                    masks=getattr(calculator.tracks,'masks',None)
                    metrics['mask_count']=len(masks) if masks is not None else 0
                last_metrics=metrics
                rows.writerow(dict(frame=frames,video_time_s=round((frames-1)/fps,4),
                                   process_ms=round(elapsed*1000,3),total_tracks=result.total_tracks,
                                   metrics_json=json.dumps(metrics,ensure_ascii=False,default=jsonable)))
                plot=result.plot_im
                cv2.putText(plot,f'{app} | {variant} | frame {frames}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
                if status and status != 'calibrated_approximation':
                    cv2.putText(plot,'DEMO / SCALE TEST - NOT VERIFIED SPEED',(12,56),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
                writer.write(plot)
                if not args.no_display:
                    cv2.imshow(window,plot)
                    paused=False
                    delay=max(20,min(100,int(1000/fps-elapsed*1000)))
                    while True:
                        key=cv2.waitKey(30 if paused else delay)&0xFF
                        if key in (ord('q'),27):stop=True;break
                        if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:stop=True;break
                        if key==ord('s'):
                            snapshots+=1;p=shots/f'frame_{frames:06d}_{snapshots:02d}.png'
                            if not cv2.imwrite(str(p),plot):raise RuntimeError('截图保存失败。')
                            print(f'已保存截图：{p}')
                        elif key==32:paused=not paused
                        if not paused:break
                if max_frames and frames>=max_frames:break
        mean_ms=statistics.mean(timings)*1000 if timings else None
        summary=dict(app=app,app_name=APPS[app]['name'],variant=variant,run_id=run_id,
                     parameters=cfg,source=str(source),source_sha256=fingerprint(source),
                     source_width=width,source_height=height,source_fps=fps,
                     frames_processed=frames,completed_video=completed,stopped_by_user=stop,
                     warmup_frames_excluded=10,mean_process_ms=mean_ms,
                     processing_fps=1000/mean_ms if mean_ms else None,
                     timing_definition='每帧检测 跟踪 业务计算与标注，不含读取 保存 显示 暂停 截图 模型下载和加载',
                     ultralytics=ultralytics.__version__,python=sys.version.split()[0],
                     torch=torch.__version__,opencv=cv2.__version__,platform=platform.platform(),
                     measurement_status=status,speed_calibration_note=settings.get('speed_calibration_note'),
                     last_metrics=last_metrics,screenshot_count=snapshots,screenshot_dir=str(shots))
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=jsonable),encoding='utf-8')
        print(f'\n结果已保存：{out}')
        if mean_ms:print(f'平均处理耗时 {mean_ms:.2f} ms/帧；处理速度 {1000/mean_ms:.2f} FPS')
        print('比较功能结果时，请使用相同素材和相同帧号；提前退出会影响累计计数，不能与全片结果直接比较。')
    finally:
        cap.release()
        if writer is not None:writer.release()
        if not args.no_display:cv2.destroyAllWindows()

def entry(default_app=None):
    try:main(default_app)
    except (FileNotFoundError,ValueError,RuntimeError) as e:
        print(f'\n操作提示：{e}',file=sys.stderr);sys.exit(1)

if __name__=='__main__':
    entry()
