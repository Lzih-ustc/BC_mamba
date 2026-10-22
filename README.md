# BC_mamba

[English](#english) | [简体中文](#中文介绍)

## English

**Binary-Compressed Selective Scans for Training-Efficient Sequence Modeling**

BC_mamba is research code for event-driven sequence modeling with sparse, binary-compressed selective scans. The accompanying manuscript calls the architecture **Dual-State Spiking Mamba (DS-Mamba)**. The repository name remains **BC_mamba**.

![BC_mamba architecture: network overview and dual-state DS-Mamba block](assets/bc_mamba_architecture.png)

*Architecture overview. The left panel shows the full sequence model; the right panel shows the event-driven scan and continuous-memory paths inside one block.*

### What the model does

DS-Mamba separates two roles that are often combined in sequence models:

- **Binary spike state at layer l**: learned threshold routing selects active channel-position events and supplies the features used by the recurrent scan. A straight-through estimator supports training through the binary threshold.
- **Continuous memory state at layer l**: residual updates accumulate restored scan outputs across network depth and provide the representation used to generate the next layer's spikes.

Inside a block, active events are packed into flat channel streams. The selective state-space recurrence runs at those events only. The elapsed time across skipped positions is preserved through inter-spike decay, and a causal zero-order hold (ZOH) restores event outputs to a dense sequence before they update the continuous memory. The scan work therefore depends on the number of active events; projections and restoration still include dense operations.

### Scope and reported results

This code package contains the model implementation and training scripts for the five Long Range Arena (LRA) tasks: ListOps, Text, Retrieval, Image, and Pathfinder. The manuscript also reports Speech Commands (SC10) and WikiText-103 experiments, but those experiment pipelines are not included in this LRA code archive.

The manuscript reports an LRA average accuracy of **70.15%** for DS-Mamba, compared with **70.17%** for the S6 reference. Across the five LRA tasks, it reports **35-60% lower per-epoch training time** and **27-52% lower peak GPU memory** than S6. These are manuscript-reported measurements; runtime and memory depend on hardware and the experiment setup.

### Repository layout

- **[spike_mamba/](spike_mamba/)**: model and selective-scan interface
- **[csrc/selective_scan/](csrc/selective_scan/)**: selective-scan CUDA kernels
- **[csrc/r_nodecay/](csrc/r_nodecay/)**: recurrent output restoration kernels
- **[csrc/extract/](csrc/extract/)**: event extraction CUDA extension
- **[configs/lra.yaml](configs/lra.yaml)**: LRA configuration
- **[scripts/run_lra.sh](scripts/run_lra.sh)**: task runner
- **[train_lra.py](train_lra.py)**: Python training entry point

### Installation

The CUDA extensions require a CUDA-enabled PyTorch installation and a working CUDA compiler toolchain.

    pip install -r requirements.txt
    bash build_extensions.sh
    python tests/test_import.py

### LRA data

Set **DATA_ROOT** to a directory with the following layout. The cifar directory is optional and can be populated by torchvision.

    DATA_ROOT/
    ├── cifar/
    ├── imdb/
    ├── listops-1000/                  # or lra_release/listops-1000/
    └── lra_release/lra_release/
        ├── pathfinder32/
        └── tsv_data/                  # AAN retrieval data

The loaders also accept the fallback paths documented in [scripts/run_lra.sh](scripts/run_lra.sh).

### Run an LRA task

    bash scripts/run_lra.sh listops "$DATA_ROOT" 0
    bash scripts/run_lra.sh text "$DATA_ROOT" 0
    bash scripts/run_lra.sh retrieval "$DATA_ROOT" 0
    bash scripts/run_lra.sh image "$DATA_ROOT" 0
    bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0

The optional fourth argument sets the output directory and the fifth sets the random seed:

    bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0 outputs/lra_runs 42

For a custom configuration, use the Python entry point directly:

    python train_lra.py --task listops --data_dir /path/to/listops-1000

Run [train_lra.py](train_lra.py) --help to see all options. Training outputs are written under outputs/ and are ignored by Git.

### Manuscript

*Dual-State Spiking Mamba: Event-Driven State Space Modeling for Efficient Sequence Learning.* TNNLS 2027 submission.



### License

The code is released under the [Apache License 2.0](LICENSE). Third-party copyright notices in derived Mamba selective-scan sources are retained.

## 中文介绍

**BC_mamba：面向高效序列建模的二值压缩选择性扫描**

BC_mamba 是一个事件驱动序列建模研究代码项目，通过稀疏的二值事件压缩选择性状态空间扫描。随附论文中将该架构称为 **Dual-State Spiking Mamba（DS-Mamba，双状态脉冲 Mamba）**；GitHub 仓库名称保持为 **BC_mamba**。

上图左侧展示完整序列网络，右侧展示单个 DS-Mamba 模块中的事件扫描路径和连续记忆路径。

### 项目在做什么

DS-Mamba 将序列建模中的两种状态分开处理：

- **第 l 层的二值脉冲状态**：通过可学习阈值选择需要计算的通道和序列位置，并提供扫描分支使用的特征。训练时使用直通估计器处理二值阈值的梯度。
- **第 l 层的连续记忆状态**：跨网络层累积恢复后的扫描输出，并用于生成下一层的脉冲。

每个模块先将活跃事件打包成通道序列，只在这些事件上执行选择性状态空间递归。被跳过位置经过的时间通过脉冲间隔衰减保留下来；随后使用因果零阶保持（ZOH）将事件输出恢复成稠密序列，再更新连续记忆。因此递归扫描的计算量随活跃事件数变化；投影和序列恢复中仍包含稠密计算。

### 代码范围与论文结果

当前代码包包含模型实现，以及 Long Range Arena（LRA）五项任务的训练脚本：ListOps、Text、Retrieval、Image 和 Pathfinder。论文还报告了 Speech Commands（SC10）和 WikiText-103 实验，但这些实验流程不包含在当前 LRA 代码包中。

论文报告的 DS-Mamba 在 LRA 上平均准确率为 **70.15%**，S6 参考结果为 **70.17%**。在五项 LRA 任务中，论文报告每轮训练时间降低 **35-60%**，峰值 GPU 显存降低 **27-52%**。这些数字来自论文中的实验测量；实际速度和显存占用会受到硬件与实验配置影响。

### 目录结构

- **spike_mamba/**：模型与选择性扫描接口
- **csrc/selective_scan/**：选择性扫描 CUDA 内核
- **csrc/r_nodecay/**：递归输出恢复内核
- **csrc/extract/**：事件提取 CUDA 扩展
- **configs/lra.yaml**：LRA 配置
- **scripts/run_lra.sh**：任务运行脚本
- **train_lra.py**：Python 训练入口

### 安装

CUDA 扩展需要安装支持 CUDA 的 PyTorch，并配置可用的 CUDA 编译工具链。

    pip install -r requirements.txt
    bash build_extensions.sh
    python tests/test_import.py

### LRA 数据目录

将 **DATA_ROOT** 设置为包含下列目录结构的数据路径。cifar 可选，缺少时可由 torchvision 下载或创建缓存。

    DATA_ROOT/
    ├── cifar/
    ├── imdb/
    ├── listops-1000/                  # 或 lra_release/listops-1000/
    └── lra_release/lra_release/
        ├── pathfinder32/
        └── tsv_data/                  # AAN 检索任务数据

数据加载器也支持 [scripts/run_lra.sh](scripts/run_lra.sh) 中说明的备用路径。

### 运行 LRA 任务

    bash scripts/run_lra.sh listops "$DATA_ROOT" 0
    bash scripts/run_lra.sh text "$DATA_ROOT" 0
    bash scripts/run_lra.sh retrieval "$DATA_ROOT" 0
    bash scripts/run_lra.sh image "$DATA_ROOT" 0
    bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0

可选的第 4 个参数指定输出目录，第 5 个参数指定随机种子。自定义配置可以直接调用 train_lra.py：

    bash scripts/run_lra.sh pathfinder "$DATA_ROOT" 0 outputs/lra_runs 42
    python train_lra.py --task listops --data_dir /path/to/listops-1000

使用 [train_lra.py](train_lra.py) --help 查看全部选项。训练输出写入 outputs/，并由 Git 忽略。

### 论文

*Dual-State Spiking Mamba: Event-Driven State Space Modeling for Efficient Sequence Learning.* TNNLS 2027 投稿稿件。

### 许可证

代码采用 [Apache License 2.0](LICENSE) 发布。派生自 Mamba selective-scan 的第三方版权声明已保留。
