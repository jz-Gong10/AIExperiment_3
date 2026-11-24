"""Summarize measured runs; never invent values for missing experiments."""
from pathlib import Path
import csv
import json
from app_specs import APPS,VARIANTS
ROOT=Path(__file__).resolve().parent
FIELDS=['app','应用','variant','对比组','运行状态','运行目录','处理帧数','全片完成','平均毫秒每帧','处理FPS','截图数','速度有效性','最终业务指标']
def main():
    rows=[]
    for app,spec in APPS.items():
        for variant,title in VARIANTS.items():
            paths=sorted((ROOT/'results'/app/variant).glob('*/summary.json'))
            row={'app':app,'应用':spec['name'],'variant':variant,'对比组':title,'运行状态':'待运行'}
            if paths:
                p=paths[-1];d=json.loads(p.read_text(encoding='utf-8'))
                row.update({'运行状态':'已记录','运行目录':str(p.parent),'处理帧数':d['frames_processed'],
                            '全片完成':d['completed_video'],'平均毫秒每帧':d['mean_process_ms'],
                            '处理FPS':d['processing_fps'],'截图数':d['screenshot_count'],
                            '速度有效性':d.get('measurement_status'),'最终业务指标':json.dumps(d['last_metrics'],ensure_ascii=False)})
            rows.append(row)
    p=ROOT/'reports'/'实验结果汇总.csv'
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print('汇总已保存：',p)
    print('每个应用每组取最近一次记录。对比前检查素材哈希 参数 帧数与完成状态是否一致。')
if __name__=='__main__':main()
