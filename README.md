# simu — 闭环、调度精确的 KV cache serving 模拟器

闭环模拟一台聚合 SGLang 服务器重放 agent 流（多轮 + 工具间隔，闭环：下一轮到达 = 上一轮完成 + 工具间隔）。
硬件压缩成每台机器建一次档的步代价速率（`profiles/machines/*.json`）；主机负载不建模——档案描述的是建档时的机器环境。

两个实现：
- **`sim/sgl_sim.py`（主实现）**：直接驱动 SGLang **真实的** `Scheduler`（CPU、虚拟时钟、真实 overlap 事件循环），只替换 GPU 层
  （假 worker 按机器档案给每步计价）、IPC 和时钟。调度规则一条不手写；ServerArgs 可从 GPU 运行日志读取（`--server-args-from`）。
- **`sim/des_sim.py`（参考实现）**：同一套机器档案与客户端模型，调度规则逐条照抄 `scheduler.py` / `schedule_policy.py`，用于交叉校验。

性能对比的 token 输入用 GPU 实际发送/生成的 token（`--ids-from ARM`）：真实分叉下的性能对应真实分叉下的输出；同一工作点不同策略的
GPU 输出基本一致（74–98%），所以"真实模型跑一次得到 token 轨迹，再模拟各策略"是合法用法（跨臂检验见 `scripts/run_cross_arm.sh`）。

2026-09-27 从 `sglang` 仓库（分支 `feat/rr-declared-recency-synthesis`）的 `results/scripts/` 独立出来；原位置的文件保留未动。

## 目录

| 路径 | 内容 |
|---|---|
| `sim/sgl_sim.py` | 主实现：真实 SGLang Scheduler + 假 worker + 虚拟时钟（文件头 docstring 写了替换了哪些部分） |
| `sim/des_sim.py` | 参考实现（文件头 docstring 写了每条手写的调度规则） |
| `sim/offline_replay.py`, `sim/replay_harbor_stream.py` | 流加载（`load_stream`）、`ArmSpec`、`SimClock`、trial 分组；也是开环 baseline（按录制时刻重放）。流 pickle 以模块名 `rhs` 写成，所以 `replay_harbor_stream.py` 必须以 `rhs` 名导入（`offline_replay.load_rhs` 负责） |
| `sim/closed_loop_sim.py` | 旧版闭环模拟器（简化准入，无 PrefillAdder/overlap/cuda-graph 语义）——作为"朴素闭环" baseline |
| `sim/steplog_parse.py` | server.log 的 Prefill/Decode 步行 → CSV |
| `sim/fit_step_costs.py` | 从 server.log 拟合步代价（`--joint`：与请求日志重建的最大上下文联合拟合） |
| `sim/profile_machine.py` | 一台机器建档：`fit` / `ttft` / `client` |
| `sim/gpu_windows.py`, `sim/sim_vs_gpu_pair.py`, `sim/des_ensemble_summary.py`, `sim/des_noise_band.py` | GPU 臂同口径指标、模拟↔GPU 逐请求配对、集合均值±sd、噪声带 |
| `profiles/machines/` | 机器档案：`pod0920.json`（09-20 云 pod，12 臂）、`local8xh100.json`（本机，35 臂）+ 对应臂清单 `.tsv` |
| `profiles/params*/` | 早期按臂拟合的参数（第一/二轮验证用，已被机器档案取代） |
| `scripts/run_pod0920_ensemble.sh` | pod0920 12 臂 × 8 噪声种子闭环集合（des_sim） |
| `scripts/run_local_sgl_ensemble.sh`, `scripts/run_local_des_ensemble.sh` | 本机档案全部 GPU 臂 × 噪声种子（`IDS=1` 用 GPU token；ServerArgs 取自各臂日志） |
| `scripts/run_cross_arm.sh` | 跨臂检验：用同工作点另一策略臂的 GPU token 模拟目标臂 |
| `scripts/ladder/` | motivation 基线阶梯图（B1 录制时刻重放 / B2 开环 DES / B3 朴素闭环 / des_sim / sgl_sim），结果 `docs/MOTIVATION_LADDER.md` |
| `analysis/` | 第五/六轮验证的一次性诊断脚本（墙窗配对、漂移分解、忙时闭合等），参数多为 GPU 臂目录，未整理 |
| `docs/VALIDATION_NOTES.md` | 六轮验证记录（对齐要点、误差链条、开环/闭环/混合拆分、机器档案） |

## 依赖

- **SGLang（本 fork）**：`sglang.srt.mem_cache.{radix_cache, rr_radix_cache, evict_policy, base_prefix_cache, cache_init_params}`。
  RR / declared / chunk 变体只在 fork 里有。目前直接用 sglang 仓库的 venv：`/home/chuxi/Project/sglang/.venv/bin/python`（editable 安装）。
- numpy、torch（CPU 即可；模拟器在 import 前把线程数钉为 1）、pandas（仅 `profile_machine.py`）。
- 本机 `sgl_kernel` 需要 `LD_LIBRARY_PATH=/extra/chuxi/libs`（libnuma）；`/tmp` 满时设 `TMPDIR`。
- 数据（录制流 `*.pkl`、GPU 臂目录）仍在 sglang 仓库的 `results/` 下，清单里的路径相对于它（`DATA_ROOT`）。

## 用法

```bash
cd /home/chuxi/Project/sglang            # DATA_ROOT：流与 GPU 臂目录的相对路径以此为根
export LD_LIBRARY_PATH=/extra/chuxi/libs:$LD_LIBRARY_PATH TMPDIR=/home/chuxi/tmp
.venv/bin/python ../simu/sim/des_sim.py results/terminus_recordings/spider2_terminus/trace_x/spider2_terminus_x2.pkl \
    --kv 750000 --conc 40 --seed 0 --rec-conc 4 --rec-span 9765.1 --trial-gap 0 \
    --arms rrdeclband0.001,rrdeclchunk1024 --sched fcfs \
    --params-json ../simu/profiles/machines/pod0920.json --noise-seed 1 --out-dir OUT --csv
```

模式：
- **预测模式**（只用流 + 机器档案）：上面的命令。
- **诊断模式**（使用 GPU 运行产物）：`--arrivals-from ARM`（开环，GPU 实测到达）、`--open-until T`（T 前开环、之后闭环，des_sim）、`--drops-from ARM`（复现客户端中途丢对话）、`--ids-from ARM`（GPU 实际 token）。
- 主机负载的建模尝试（`--load-from`、按执行模式的负载项、档案 v2）保存在分支 `load-modeling`，不在 master 上。

sgl_sim 用法（在 DATA_ROOT 下）：
```bash
.venv/bin/python ../simu/sim/sgl_sim.py STREAM.pkl --kv 750000 --conc 46 --seed 36 --rec-conc 4 --rec-span 27366.0 --trial-gap 366.2 \
    --arm rrdeclchunk1024 --params-json ../simu/profiles/machines/local8xh100.json --noise-seed 1 \
    --server-args-from GPU_ARM --ids-from GPU_ARM --out-dir OUT --csv
```

资源（用户规则）：CPU 作业总计 ≤30 线程、≤100 GB。单进程 1 核 / ~4.8 GB（`--ids-from` 10–23 GB，banking 最大）→ 集合并发默认 12（`--ids-from` ≤4）。sgl_sim 一臂（~77 min 虚拟时间）约 70–100 s。

## 独立后的验证

2026-09-27：原位置与本目录各跑同一臂（spider2 x2 c40 s0，pod0920 档案，noise-seed 1）——请求/步/占用 CSV 逐字节相同；`summary.json` 只有 `cpu.evict@k`（模拟器自身逐出调用的墙钟耗时，诊断量）不同。

## 已改动（相对原文件）

- `sim/offline_replay.py`：`RHS` 指向同目录的 `replay_harbor_stream.py`（原为 `bench/scripts/`）。
- `analysis/*.py`：`sys.path` 与档案路径改为相对本目录。
- `scripts/run_pod0920_ensemble.sh`：`SIMU` / `DATA_ROOT` / `PY` 可配置。
