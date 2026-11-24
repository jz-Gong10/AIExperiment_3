# DCU 热力图与目标计数实验

将实验二的热力图和目标计数迁移到已经配置好 DCU PyTorch 的 Docker 容器。
仓库包含代码、篮球视频及 YOLO26 n/s 检测权重，可直接下载整个仓库。
本仓库对应本次课程的“八选二”部署部分；配电箱数据集训练与验证需另外准备。

## 1 下载到指定目录

以下命令在已经配置好的 Docker 容器里执行。如果目前在宿主机上，先用
`docker ps` 查找容器，再运行 `docker exec -it 容器名 /bin/bash`。
如果队友使用 Conda，进入容器后还需激活她配置好的同一个环境。
若当前 Python 已通过 `check_server.py` 的依赖和设备运算检查，可直接使用当前环境，
无需额外激活 Conda。容器的默认 Python 本身也可以是已配置好的实验环境。

```bash
mkdir -p /workspace/gjz_3
curl -fL --retry 3 https://github.com/jz-Gong10/AIExperiment_3/archive/refs/heads/main.tar.gz -o /tmp/gjz_3.tar.gz
tar --keep-old-files -xzf /tmp/gjz_3.tar.gz --strip-components=1 -C /workspace/gjz_3
cd /workspace/gjz_3
```

解压后的入口直接位于 `/workspace/gjz_3/02_heatmap.py` 和
`/workspace/gjz_3/03_counting.py`，不会额外套一层仓库目录。
`--keep-old-files` 会拒绝覆盖已有文件，以上适用于首次下载。若目录已有同名文件，
先确认其中是否有自己的修改，再决定如何更新。
若容器中只有 `python3` 命令，后面的 `python` 改成 `python3`；执行批量脚本时
使用 `PYTHON_CMD=python3 bash run_server.sh smoke`。

也可以用 Git 下载，和上面的下载解压方式任选一个：

```bash
git clone https://github.com/jz-Gong10/AIExperiment_3.git /workspace/gjz_3
cd /workspace/gjz_3
```

## 2 检查已经配置好的环境

```bash
python check_server.py
```

脚本会检查依赖导入、两个 Solutions 接口、设备 0 上的实际矩阵运算、视频读取，
并核对视频与权重的 SHA256。结果保存到 `reports/environment_check.json`。
海光 DCU PyTorch 同样使用 `torch.cuda` 接口；不能仅凭这个接口名称判断设备品牌。
完整业务兼容性还需下一步的 50 帧试运行确认。

当前应用以 Ultralytics 8.4.176 为参照。若缺少 Ultralytics，可仅安装应用包：

```bash
python -m pip install --no-deps -r requirements-apps.txt
python check_server.py
```

`--no-deps` 不会安装传递依赖。若检查仍提示缺包，应按实际缺失补齐。
若报告二进制库、Python 版本或算子不兼容，需要依据服务器环境处理；
不要用本机实验二的 requirements.txt 覆盖队友配置好的 DCU torch/torchvision。
批量脚本关闭了 Ultralytics 自动安装依赖，避免运行时自动更改共享环境。

## 3 先试运行再完整运行

每个应用先运行 50 帧：

```bash
bash run_server.sh smoke
```

两项应用都通过后，完整运行两项原始参数实验：

```bash
bash run_server.sh full
```

如需沿用实验二的四组对比，每个应用运行原始参数、参数一、参数二和 s 模型，
总共八组，执行：

```bash
bash run_server.sh all
```

以上各批次依次运行，不会同时启动多项实验。同一张 DCU 的正式训练请和队友协调时间。
不要同时执行 full 和 all；all 已包含 full 中的原始参数组。

也可以单独运行：

```bash
python 02_heatmap.py --variant baseline --no-display
python 03_counting.py --variant baseline --no-display
```

`settings.json` 默认 `device` 为 `"0"`。有多张可见设备时可指定
`EXPERIMENT_DEVICE=1 bash run_server.sh smoke`。本机 CPU 检查可使用
`EXPERIMENT_DEVICE=cpu bash run_server.sh smoke`，但不能将它报告为 DCU 实验。

## 4 查看结果

- `results/heatmap/组别/时间戳/output.mp4`：热力图视频。
- `results/counting/组别/时间戳/output.mp4`：目标计数视频。
- 每次运行的 `metrics.csv`：逐帧处理耗时、轨迹与业务指标。
- 每次运行的 `summary.json`：实际参数、环境版本、输入哈希、处理帧数、完成状态及平均性能。
- `reports/实验结果汇总.csv`：每个应用每组最近一次运行的汇总。

服务器默认关闭窗口，`--no-display` 不产生按键截图；请保存实际运行的终端截图，
并下载结果视频查看效果。只有完整处理或相同帧范围的累计计数才适合横向比较。
50 帧 smoke 结果仅用于检查。再次汇总可执行 `python collect_results.py`。

## 报告截图与已有结果图

在项目目录执行以下命令，分别截图。它们读取已有记录，不会训练模型：

```bash
python report_capture.py env
python report_capture.py data
python report_capture.py apps
python report_capture.py train --run smoke_n
```

`env` 显示此前保存的环境检查；`data` 显示本项目 `data_box.yaml` 和数据文件数；
`apps` 选取两项应用最近一次完整运行，无完整运行时明确显示部分运行；
`train` 显示指定训练目录的配置与末轮 CSV 指标。末轮指标不一定对应最优权重；
推理耗时应以真实验证输出为准。

收集本次应用视频的同帧结果图片和已有训练图，可执行：

```bash
python report_capture.py figures --run smoke_n
```

它从两段真实输出视频提取同一帧，复制训练目录中已有的结果图和参数文件，
生成 `report_materials/` 与 `report_materials.zip`。不会重新推理、训练或弹出窗口。
默认是第 1000 帧，若可用帧数不足则使用两组都存在的帧。
将压缩包传回本地后可查看原图并插入报告。正式训练完成后，
将 `--run smoke_n` 换成实际正式实验目录，分别收集每组结果。

## 文件来源

`data/general.mp4` 为本地实验二已有篮球视频。
两个模型来自 Ultralytics 官方 assets 发布资源；大小、SHA256 和来源 URL 见
`assets_manifest.json`。原始 Windows 实验目录保持原样。
