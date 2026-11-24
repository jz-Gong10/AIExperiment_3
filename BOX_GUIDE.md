# 配电箱训练与验证操作说明

已完成的 smoke_n 是 1 个 epoch 的流程检查。下列四组是独立的正式对比，
全部从原始预训练权重开始，不使用 smoke_n 的 best.pt 接着训练。
在已经验证可用的同一个 Docker 容器及 Python 环境里运行，不重新安装依赖。
项目目录是 `/workspace/gjz_3`，输入配置是之前创建的 `data_box.yaml`。

## 下载脚本

```bash
cd /workspace/gjz_3
curl -fL --retry 3 https://raw.githubusercontent.com/jz-Gong10/AIExperiment_3/main/box_experiment.py -o box_experiment.py
curl -fL --retry 3 https://raw.githubusercontent.com/jz-Gong10/AIExperiment_3/main/BOX_GUIDE.md -o BOX_GUIDE.md
```

## 保存并查看对比方案

```bash
python box_experiment.py plan --epochs 10 --batch 16
```

该命令读取安装版本的四项默认超参数，检查单类别 YOLO 标签，记录训练/验证集的
图片数、背景图片数和目标数，将方案保存到 `box_protocol.json`。同一方案创建后，
再次执行 plan 仅显示原方案，不会覆盖设置。四组都必须使用同一方案。
数据签名包括图片路径/大小/修改时间和完整标签内容，不是全部图片内容的 SHA256。

|组名|预训练模型|超参数配置|
|---|---|---|
|n_base|yolo26n.pt|四项使用当前版本默认值|
|n_tuned|yolo26n.pt|修改四项|
|s_base|yolo26s.pt|四项使用当前版本默认值|
|s_tuned|yolo26s.pt|与 n_tuned 相同的修改|

以当前参考版本为例，修改 `lr0: 0.01 → 0.005`、`lrf: 0.01 → 0.1`、
`weight_decay: 0.0005 → 0.001`、`warmup_epochs: 3.0 → 1.0`。
以 plan 实际打印的默认值为准。优化器统一为 SGD，避免 optimizer=auto 忽略学习率。

公共条件：10 个 epoch、imgsz=640、batch=16、workers=0、amp=False、seed=0、
patience=0、close_mosaic=0。这里“基准”指四项待比较超参数使用默认值，
并非所有训练设置都为默认；close_mosaic=0 表示这个短周期实验不在末尾关闭 mosaic。
四项同时修改能比较组合效果，不能据此断言某一项单独造成指标变化。

10 轮是首轮对比预算，不能预先声称收敛。若曲线未稳定或效果需改善，
后续应在同一新方案下给四组统一增加轮数，重新比较。

## 依次训练

与队友协调 DCU 空闲时段后，一次只执行一组：

```bash
python box_experiment.py train --group n_base
python box_experiment.py train --group n_tuned
python box_experiment.py train --group s_base
python box_experiment.py train --group s_tuned
```

每组都打印实际进度，结束后出现 `TRAINING COMPLETE: 组名`。截图保存训练完成、
轮数、权重保存路径。输出在 `runs_box/组名/`，其中有 best.pt、args.yaml、results.csv、
曲线、验证预测图以及完成记录。所有原始结果继续保留。

如果想一次顺序运行全部四组并让关闭网页后继续：

```bash
mkdir -p logs
nohup python -u box_experiment.py train --group all > logs/box_train_all.log 2>&1 < /dev/null &
```

查看进度：

```bash
tail -n 40 logs/box_train_all.log
```

动态查看可执行 `tail -f logs/box_train_all.log`，Ctrl+C 只退出 tail。
不要再同时执行前台的训练命令。nohup 可避免终端退出引起的挂断，但容器停止、
服务器重启或任务报错仍会终止训练。基于先前 batch=4 的试训练耗时，
完整四组可能需要数小时；batch=16 的实际耗时应以第一组进度为准。

同一命令重复运行会跳过已有完成记录的组。若中断且已产生 last.pt，可使用：

```bash
python box_experiment.py train --group all --resume
```

无 last.pt 的部分目录或参数不匹配时会停止，防止悄悄覆盖结果。
若出现显存不足，先保留报错，统一重设实验方案后再比较；不能只改变其中一组 batch。

## 用 val 方法统一评估

四组训练都完成后执行：

```bash
python box_experiment.py validate
```

脚本逐一加载各组 best.pt，并显式调用 model.val()。验证条件统一为相同 val 数据、
imgsz=640、batch=1、FP32、conf=0.001、iou=0.7、max_det=300 和同一设备。
每组保存 Precision、Recall、mAP50、mAP50-95、单张推理耗时五项。
推理耗时由 val() 返回，不包含预处理与后处理；同组内数据较少时计时有波动。

每组验证后出现 `VALIDATION COMPLETE`，记录在 `runs_box/组名/validation.json`，
曲线、混淆矩阵和验证预测图保存在 `box_validation/` 中。
需要重跑验证时执行 `python box_experiment.py validate --repeat`。

## 截图和打包

查看最终五项指标表并截图：

```bash
python box_experiment.py table
```

表中 Precision、Recall、mAP 是 0 到 1 的小数。原始汇总表另存为
`reports/box_comparison.csv`。保留 plan 参数表、各组训练结束和验证输出、最终指标表截图。
训练曲线和预测效果图直接用原图即可。

打包四组代码、参数、原始结果图、验证结果和 best.pt：

```bash
python box_experiment.py pack
```

得到 `/workspace/gjz_3/box_report_materials.zip`。
若已有前一阶段的 report_materials，也会放入同一包；不包含课程原始数据集或整个环境。
如继续通过自己的 GitHub Release 传回本机：

```bash
python upload_report.py box_report_materials.zip
```

按脚本提示输入具有本仓库 Contents 写权限的临时令牌（隐藏输入，不要发给他人）。
公开仓库的 Release 附件也公开。下载到 `D:\桌面\人工智能综合实践1\3`，
解压后用真实指标与图片完成报告；至少记录两个实际问题及解决方法。
