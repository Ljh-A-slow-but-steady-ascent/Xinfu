# 信肤 · Xinfu 操作手册

信肤是一套皮肤科辅助工作台，提供病例管理、病灶分析、医学检索、报告编辑和归档。系统架构与训练实验见[《项目介绍》](./项目介绍.md)。

代码：[GitHub](https://github.com/Ljh-A-slow-but-steady-ascent/Jiderma) · 任务权重：[Hugging Face / Xinfu-Adapters](https://huggingface.co/Ljh-hgf/Xinfu-Adapters)

**在线访问：[信肤公网工作台](https://xinfu.tail7dc076.ts.net/)**。使用项目账号登录，浏览器即可访问。

首次安装从[第 8 节](#8-首次部署与配置)开始。模型下载命令按[版本清单](./models/manifest.json)准备文件并校验 SHA-256。

## 1. 启动与登录

在项目目录执行：

```bash
bash scripts/run_web.sh
```

打开[信肤工作台](http://127.0.0.1:8501/)，使用管理员配置的账号登录。终端显示访问地址和日志位置，按 `Ctrl+C` 停止服务。已有服务运行时，直接打开现有工作台。

| 入口 | 地址或位置 |
| --- | --- |
| 网页工作台 | http://127.0.0.1:8501/ |
| API 文档 | http://127.0.0.1:8000/docs |
| 服务状态 | http://127.0.0.1:8000/health |
| 后端日志 | `artifacts/dermaguard_backend.log` |
| 前端日志 | `artifacts/xinfu_react.log` |

首次部署、账号初始化和模型配置见[第 8 节](#8-首次部署与配置)。

### 公网评审入口

公网地址：[信肤工作台](https://xinfu.tail7dc076.ts.net/)。浏览器访问后使用项目账号登录，访客无需安装 Tailscale。模型在本机 GPU 上推理。

常驻部署通过 `8501` 端口提供网页与接口，状态地址为 `http://127.0.0.1:8501/health`。systemd 管理应用、模型预热、隧道和防休眠，关闭终端后继续运行，开机自动启动。访问期间保持电脑供电、联网。

```bash
# 安装或更新常驻配置，并构建网页
bash scripts/deploy_web.sh install
# 启动工作台与本地模型
bash scripts/deploy_web.sh start
# 首次登录 Tailscale，并启用固定公网入口
bash scripts/deploy_web.sh connect
# 创建独立评审账号，已有账号文件时复用
bash scripts/deploy_web.sh account
# 查看应用和隧道状态
bash scripts/deploy_web.sh status
```

首次连接时，按终端链接登录并授权 `xinfu` 设备，再启用 Funnel 与 HTTPS。使用免费 Personal 方案。评审账号文件位于 `~/.local/share/xinfu/review-account.json`。

维护时使用 `systemctl --user restart xinfu-web` 重启应用，使用 `journalctl --user -u xinfu-web -n 50` 查看应用日志。`bash scripts/deploy_web.sh stop` 会关闭公网入口并停止工作台，随后可用 `start` 与 `connect` 恢复。

正式服务使用构建后的网页，修改前端后执行 `install` 更新构建；修改后端后重启应用。图片分析使用共享常驻进程并限制并发，页面通过后台任务查询结果；未完成的增强分析任务在应用重启后需要重新提交。

#### 公网入口维护

| 项目 | 说明 |
| --- | --- |
| 地址 | 保留设备名 `xinfu` 和当前 tailnet 域名，入口通常保持不变；日常重启、换公网 IP 和更新代码不需要换链接 |
| 在线条件 | 电脑关机、断网、休眠，或应用、隧道停止时，网页无法访问；当前已配置自动启动、故障重启与运行期间防休眠 |
| 传输 | 使用 HTTPS；Funnel 有不可自行调整的带宽限制，大文件和集中访问会受链路与本机性能影响 |
| 设备授权 | 2026-11-08 检查时，到期时间为 **2027-05-07 15:08（北京时间）**，到期前需重新授权 |
| 地址变更 | 修改设备名、重新生成 tailnet DNS 名，或删除设备后重新建机，可能需要更新入口 |

[Funnel 官方说明](https://tailscale.com/docs/features/tailscale-funnel) · [设备密钥到期](https://tailscale.com/docs/features/access-control/key-expiry) · [设备与域名](https://tailscale.com/docs/concepts/tailnet-name)。Funnel 当前仍处于 Beta。国内访问还会受运营商到中转节点的链路影响；评审前用手机流量实际打开页面并完成一次图文分析。

## 2. 认识工作台

左侧管理对话和业务资料，中间完成提问、分析与确认。

| 区域 | 用途 |
| --- | --- |
| 新对话、最近对话 | 创建、搜索、切换、重命名和删除会话 |
| 病例库 | 查看病例、病灶图片、分析结果，编辑就诊信息 |
| 报告库 | 打开报告、编辑正文、保存版本和下载 |
| 文档库 | 管理上传资料及 Agent 创建的工作文档 |
| 资料与研究 | 浏览知识条目、领域分布、训练记录和样本预测 |
| 病灶分析 | 运行图片、文本或图文比对，保存分析记录并继续追问 |

对话输入框按 `Enter` 发送，按 `Shift + Enter` 换行。右下方可选择当前可用的助手，任务卡中的“查看步骤”用于查看执行状态和工具动作。

### 三种使用模式

| 模式 | 适用任务 |
| --- | --- |
| 标准 | 日常问答、写作、编程、医学资料查询和按需联网 |
| 增强 | 从病灶图片进入快速分析，查看候选结果与相关资料 |
| 深度 | 围绕病例、问题或选定分析记录，进一步核对医学证据 |

标准模式按问题选择直接回答、联网搜索或读取网页。深度模式核对医学来源与引用。三种模式共用当前对话。

## 3. 完成一次病灶分析

### 快速获取候选与资料

1. 点击输入框旁的“＋”，选择“皮肤病灶图片”；界面自动切换到增强模式。
2. 输入病灶部位、持续时间、变化和希望核对的问题，点击发送。
3. 打开分析记录，查看图片、候选排序和医学资料。
4. 点击“加入当前对话”继续讨论，或点击“深度推理”围绕该记录继续取证。

也可以从左侧“病灶分析”进入完整面板：

| 分析方式 | 操作 |
| --- | --- |
| 图文比对 | 上传图片，与预设的十种皮肤病候选描述进行匹配 |
| 图文分析 | 提交图片和描述，查看分类候选及围绕描述检索的医学资料 |
| 图片分析 | 上传图片，查看图像分支候选 |
| 文字分析 | 输入描述，查看文本分支结果及相关资料 |

描述用于医学检索，候选按匹配程度排序。需要保存患者信息、病例编号和风险等级时，按下一节创建病例。

### 用标准模式处理普通任务

可以直接输入：

> 请联网查中国国家博物馆官网的开放时间和停止入馆时间，简短回答并附来源。

> 教我用 Python 读取 CSV，并给出一个可以运行的例子。

按对话提示补充地点、日期等查询条件。

## 4. 创建正式病例并分析

正式建档会产生病例编号，并将图片、结构化信息和分析结果关联保存。

1. 新建对话，保持“标准”模式。
2. 点击“＋”→“常规图片”附上病灶图像，并发送完整的建档描述。
3. 核对系统提取的患者信息和临床字段；如提示缺项，补充后再次附上原图提交。
4. 在操作预览卡中确认创建并分析。
5. 完成后点击“打开病例”或进入“病例库”，查看图片、Top-3 候选、风险等级、医学依据和照护路径。

可按以下格式组织输入：

> 创建病例并分析：患者陈某，男，67 岁，门诊号 OP001。左前臂病灶约 10×8 毫米，持续三个月，近两周增大并出现颜色变化，偶有出血，无瘙痒、无疼痛。无皮肤癌史、无其他肿瘤史，无过敏史。我已附上病灶图片。

| 信息 | 建档时的填写方式 |
| --- | --- |
| 身份与就诊 | 姓名、性别、年龄、门诊号 |
| 病灶描述 | 部位、主要表现、持续时间、近期变化 |
| 临床字段 | 瘙痒、疼痛、出血、增大、外观变化、两个方向的直径 |
| 相关病史 | 皮肤肿瘤史、其他肿瘤史；联系方式、过敏史等可补充 |

图片支持 JPG、PNG、WebP，单张上限 10 MB。也可先上传含病例信息的文本文档，再附图创建病例。

病例结果中的风险等级包括重新拍摄、基层复核、转诊和优先转诊。需要修改信息时，在病例详情编辑，或在关联对话中提出修改要求并核对操作预览。

## 5. 继续取证、生成报告与归档

### 围绕当前病例追问

在已关联病例的会话中继续输入：

> 根据当前病例，查找支持和不支持主要候选的依据，列出需要进一步核对的检查与转诊信息。

在分析记录中点击“加入当前对话”或“选用分析记录”，随后切换深度模式继续查询。

### 生成和编辑报告

1. 在病例关联对话中输入“根据当前病例和医学依据生成报告”。
2. 核对病例编号及操作预览，确认生成。
3. 点击返回消息中的报告入口，打开在线编辑器。
4. 编辑正文，切换预览，保存版本或下载 Markdown 文件。

报告包含患者信息、病灶图片、主诉、分析结果、风险解释、医学依据、建议及随访记录，可在编辑器中修改。

### 管理资料

- 在病例库、报告库和文档库中按关键词查找，使用文件夹分类和移动条目。
- 文档库支持上传和编辑 Markdown、TXT、JSON、CSV 文本，单份上限 2 MB。
- 可让 Agent 创建、查找、读取和改写工作文档，生成的文档可在对话中打开。
- 删除病例需输入完整病例编号，会同时清理该病例的图片、分析及报告；删除报告或工作文档会保留其关联病例。

## 6. 使用资料与研究

### 资料总览与内容

“资料总览”按十个疾病领域展示知识片段统计及分布。“内容”页可按疾病、主题、来源或正文关键词检索，打开条目后阅读片段和来源原文。

点击“上传资料”提交 MD、TXT、JSON、CSV、JPG、PNG 或 WebP，单份上限 5 MB。上传文件进入审核队列；“刷新”重新读取当前目录和队列统计。

### 训练观察与样本复核

1. 打开“训练观察”，切换仅图像、仅临床文本、图像与文本融合三种模式。
2. 查看测试集指标、训练曲线、混淆矩阵和逐类别 F1。
3. 在“结构与表示”中切换权重、激活、样本表示和结构视图。
4. 在“样本复核”中按预测正确、预测错误或类别筛选，点击图片查看病理标签、模型预测、Top-3 和临床文本。

“回放训练”用于浏览已有训练过程；重新训练使用[第 9 节](#9-训练推理与开发验证)中的训练入口。

## 7. 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 网页无法打开 | 查看启动终端，确认两个服务地址和日志；检查端口是否被其他程序占用 |
| 已打开网页，但内容读取失败 | 检查后端 `/health` 与前端代理连接，查看后端日志 |
| 登录提示失效 | 重新登录后继续操作 |
| 修改 `.env` 后账号密码没有变化 | 初始化只创建新账号；已有账号使用数据库中保存的口令 |
| 对话列表没有可用助手 | 在 `.env` 配置所选服务的密钥，重启后端后重新进入工作台 |
| 图片分析失败 | 核对模型环境路径、基础权重和任务权重，检查图片格式及大小 |
| 建档提示缺少临床信息 | 按提示补充描述，再附上原图提交 |
| 联网查询暂时没有结果 | 补充必要条件，或指定希望核对的公开网址与来源 |

## 8. 首次部署与配置

### 环境与依赖

```bash
git clone https://github.com/Ljh-A-slow-but-steady-ascent/Jiderma.git
cd Jiderma
```

| 环境 | 用途 |
| --- | --- |
| `dermaguard`，Python 3.11 | FastAPI、Agent、MCP、数据库及 Node.js 前端工具 |
| `dermfm-zero`，Python 3.9 | 本地多模态模型、医学嵌入和训练推理 |

创建两个 Conda 环境：

```bash
conda create -n dermaguard python=3.11
conda create -n dermfm-zero python=3.9
```

安装依赖：

```bash
conda run -n dermaguard python -m pip install -r requirements.txt
conda run -n dermfm-zero python -m pip install -r requirements.txt
conda install -n dermaguard -c conda-forge nodejs=22
conda run --no-capture-output -n dermaguard npm --prefix frontend ci
```

以下命令在项目根目录执行。启动入口自动定位这两个 Conda 环境；环境位于其他目录时，可通过 `XINFU_ENV_BIN` 和 `DERMAGUARD_DERMFM_PYTHON` 指定。

### 初始化配置

将 `.env.example` 复制为 `.env`，填写账号和所选对话服务的密钥。已有 `.env` 时直接编辑。

| 配置 | 作用 |
| --- | --- |
| `XINFU_ADMIN_USERNAME`、`XINFU_ADMIN_PASSWORD`、`XINFU_ADMIN_NAME` | 首次创建管理员账号 |
| `DERMAGUARD_CHAT_PROVIDER` | 选择对话服务 |
| 所选服务的 `*_API_KEY`、`*_MODEL` | 对话、查询改写和报告生成 |
| `DERMAGUARD_VISION_PYTHON` | 本地视觉模型 Python 路径 |
| `DERMAGUARD_RAG_PYTHON`、`DERMAGUARD_EMBEDDING_PATH` | 医学语义检索环境与权重 |
| `DERMAGUARD_RAG_ADVANCED=1` | 使用 BM25 与医学向量混合检索 |
| `DERMAGUARD_DATABASE_URL` | 数据库位置，默认 `sqlite:///data/dermaguard.db` |

对话服务可选 DeepSeek、智谱或 Groq，在 `.env` 对应区域填写密钥和模型名称。

### 模型与数据位置

#### 下载推理权重

本项目的独立任务权重已发布到 [Ljh-hgf/Xinfu-Adapters](https://huggingface.co/Ljh-hgf/Xinfu-Adapters/tree/main)，共 4 个文件、约 61 MB。运行时使用三种 PAD 任务头；HAM10000 线性头用于表征对照实验。

先在 [DermFM 官方模型页](https://huggingface.co/redlessone/DermFM-Zero)申请访问并接受其条款，然后在本机登录一次：

```bash
conda run --no-capture-output -n dermfm-zero huggingface-cli login
```

下载并检查全部模型文件：

```bash
conda run --no-capture-output -n dermfm-zero python scripts/download_models.py
conda run --no-capture-output -n dermfm-zero python scripts/download_models.py --verify
```

只获取本项目任务头时，使用 `--only adapters`，无需登录。下载中断后重新执行即可继续；已经通过校验的文件会直接复用。全部资源约 1.88 GB，具体版本、文件大小和哈希见 [models/manifest.json](./models/manifest.json)。

| 获取来源 | 项目中的存放位置 | 用途 |
| --- | --- | --- |
| [信肤任务权重](https://huggingface.co/Ljh-hgf/Xinfu-Adapters) | `artifacts/training/pad_multimodal/image_only/best.pt` | 图像分类头 |
| 同上 | `artifacts/training/pad_multimodal/text_only/best.pt` | 文本投影与分类头 |
| 同上 | `artifacts/training/pad_multimodal/image_text/best.pt` | 图文融合模块与分类头 |
| 同上 | `artifacts/training/ham7_linear_head/ham7_linear_head.pt` | HAM10000 线性分类头 |
| [DermFM-Zero 官方仓库](https://huggingface.co/redlessone/DermFM-Zero) | `weights/DermFM-Zero-Open/open_clip_pytorch_model.bin`、同目录配置 | 专业皮肤图文基础权重 |
| [PubMedBERT 官方仓库](https://huggingface.co/neuml/pubmedbert-base-embeddings) | `weights/text/PubMedBERT/` | 本地文本配置与分词器 |
| [IlhaEmbed 官方仓库](https://huggingface.co/weemed/IlhaEmbed) | `weights/rag/IlhaEmbed/` | 医学语义嵌入模型 |

DermFM 基础权重按官方访问条款下载。完整检查点已含文本塔参数，PubMedBERT 目录保存配置和分词器。图像推理和医学嵌入均在本地完成。

#### 下载数据集

| 数据集 | 下载与来源 | 项目使用方式 |
| --- | --- | --- |
| PAD-UFES-20 | [原始数据与说明](https://data.mendeley.com/datasets/zr7vgbcyr2/1) · [作者实验数据包](https://drive.google.com/drive/folders/1lH53xfkdkBLSMDn6WShjCRV9gi7xQTvM?usp=drive_link) | 图文训练、三种输入对比、训练观察中的图片与临床信息 |
| HAM10000 | [原始数据与说明](https://doi.org/10.7910/DVN/DBW86T) · [作者实验数据包](https://drive.google.com/drive/folders/1lH53xfkdkBLSMDn6WShjCRV9gi7xQTvM?usp=drive_link) | 冻结图像表征的七分类线性探针 |
| 医学检索资料 | `backend/app/knowledge/rag_documents/` | 随代码提供的来源、章节和知识片段 |

**PAD：**从作者数据目录下载 `multimodal_finetune.zip`，存到 `data/downloads/`，执行：

```bash
mkdir -p data/downloads
# 将下载的 multimodal_finetune.zip 放入 data/downloads/ 后执行
conda run -n dermfm-zero python scripts/prepare_pad_ufes20.py \
  --archive data/downloads/multimodal_finetune.zip
```

该命令提取 `multimodal_finetune/PAD/`，保存原始 CSV，并生成使用当前机器路径的训练 CSV。作者的 train / val / test 划分对应 1,838 / 230 / 230 张图片。

**HAM10000：**从同一作者数据目录下载 `linear_probe.zip`，将压缩包中的 `linear_probe/` 解压到 `third_party/DermFM-Zero/data/`，保留 `HAM-official-7-lp/meta.csv` 中的划分和路径。

完成后，目录应为：

```text
data/
├── downloads/multimodal_finetune.zip
└── pad_ufes20/
    ├── images/                         # 2,298 张原始 PNG
    ├── official/meta/
    │   ├── metadata.csv
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    ├── meta/{train,val,test}.csv        # 当前机器的图片路径
    └── split_audit.json
third_party/DermFM-Zero/data/linear_probe/HAM-official-7-lp/
├── images/                             # HAM10000 图像
└── meta.csv                            # image_path,label,split,diag
```

使用自带图片推理只需模型文件；重新训练和样本复核需下载对应数据集。PAD 原始数据采用 CC BY 4.0；HAM10000 与作者数据包的使用条件见下载页面。

### 启停与迁移

```bash
# 不自动打开浏览器
DERMAGUARD_NO_BROWSER=1 bash scripts/run_web.sh

# 指定端口
DERMAGUARD_API_PORT=8002 DERMAGUARD_UI_PORT=8502 bash scripts/run_web.sh

# 启动独立 MCP stdio 服务
bash scripts/run_mcp_server.sh
```

独立 MCP 入口使用标准输入输出接收客户端请求。迁移机器时重新执行模型下载与 PAD 数据准备命令；启动入口会重新定位环境，`.env` 中手动指定的环境路径需同步调整。

业务备份包含 `data/dermaguard.db`、`data/cases/` 和 `data/knowledge_uploads/`。停止服务后统一复制数据库和对应文件；模型权重与训练结果单独保存。

## 9. 训练、推理与开发验证

训练和推理使用 `dermfm-zero` 环境。

```bash
# 查看数据准备参数
conda run -n dermfm-zero python scripts/prepare_pad_ufes20.py --help

# 以相同数据划分运行三种输入模式
bash scripts/train_pad.sh --mode all --epochs 30 --batch-size 4 \
  --accumulation-steps 8 --precision bf16 --device cuda

# 单样本推理
bash scripts/infer_pad.sh \
  --image data/pad_ufes20/images/PAT_980_1849_295.png \
  --metadata examples/pad_patient_example.json \
  --model-checkpoint artifacts/training/pad_multimodal/image_text/best.pt \
  --device cuda

# 重绘已有 PAD 训练结果
conda run -n dermfm-zero python scripts/visualize_pad_training_results.py

# 查看 HAM10000 实验入口
conda run -n dermfm-zero python -m experiments.ham10000.train --help
conda run -n dermfm-zero python -m experiments.ham10000.plot --help
```

```bash
# 后端回归
DERMAGUARD_CHAT_PROVIDER=mock DERMAGUARD_RAG_ADVANCED=0 \
  conda run -n dermaguard python -m pytest backend/tests -q

# 多模态模块测试
conda run -n dermfm-zero python -m pytest tests/test_pad_multimodal.py -q

# 前端检查与构建
conda run --no-capture-output -n dermaguard npm --prefix frontend run build
conda run --no-capture-output -n dermaguard node --test \
  frontend/src/api.test.js frontend/src/workspace.test.js
```

## 10. 代码导航

| 位置 | 职责 |
| --- | --- |
| `frontend/src/App.jsx` | 会话、病例、报告、文档及模式交互 |
| `frontend/src/Login.jsx`、`KnowledgeCenter.jsx` | 登录与资料、模型观测界面 |
| `backend/app/api/` | 按业务拆分的接口与输入输出模型 |
| `backend/app/harness/` | Agent 工具循环、任务状态和确认流程 |
| `backend/app/services/` | 分析、报告、检索与隐私服务 |
| `backend/app/services/rag/` | 查询构造、混合召回、证据工具与引用核对 |
| `backend/app/mcp/` | 统一工具注册和 MCP 服务 |
| `backend/app/database/` | 用户、病例、报告、文档和任务记录 |
| `pad_multimodal/`、`experiments/ham10000/` | 多模态训练推理及视觉对照实验 |
| `models/manifest.json`、`scripts/download_models.py` | 模型固定版本、下载路径和文件完整性校验 |

项目代码采用 [MIT License](./LICENSE)；`third_party/DermFM-Zero/` 保留其原始许可，模型和数据的使用条件分别见对应来源页面。

接口参数与交互格式见[本地 API 文档](http://127.0.0.1:8000/docs)。
