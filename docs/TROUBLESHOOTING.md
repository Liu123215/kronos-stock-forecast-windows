# 排错与手动放置权重

## 先保留完整信息

记录应用页脚版本、选择的模型、输入窗口、股票代码、行情截至日期和完整报错。安装包完整解压后再运行；升级时关闭旧窗口。

不要为了运行而关闭 SHA256 校验，不需要把 API Key、HF 令牌、密码或个人券商资料上传到 Issue。

## 文件与 Python

**找不到 requirements.txt**：源码方式应在仓库根目录运行。Windows 启动包只需要双击对应 CMD；不要在 ZIP 预览窗口中运行。

**找不到 .venv 中的 Python**：先创建环境。可以不激活，直接使用其 Python 完整相对路径：

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe --version
~~~

一键启动器管理自己的独立环境，不依赖手工建立的 .venv。

## 网络超时

START_KRONOS.cmd 使用官方站点。START_KRONOS_MIRROR.cmd 使用 hf-mirror.com 第三方公益镜像，二者不会自动互相切换。

程序设置较长超时，最多重试 3 次并保留可续传缓存。等待更长不能修复完全不可达的网络。切换入口时先关闭旧窗口，重启后核对页脚显示的实际下载来源。

## 手动下载

默认目标目录为 C:\AI\Kronos\model_files。使用备用安装位置时，以窗口显示的 **Model files folder** 为准。每个组件目录内放置 model.safetensors，配置由程序准备。

| 使用的模型 | 必需组件 |
| --- | --- |
| Mini | Kronos-Tokenizer-2k + Kronos-mini |
| Small | Kronos-Tokenizer-base + Kronos-small |
| Base | Kronos-Tokenizer-base + Kronos-base |

固定官方下载链接：

- [Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k/resolve/b22fb9cb30a2de2f77e8b617169cd756ba964a08/model.safetensors)
- [Mini](https://huggingface.co/NeoQuasar/Kronos-mini/resolve/7fdcc628d87f325ccdbcae0a372622ca7e6813aa/model.safetensors)
- [Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/resolve/9ef143b98ee3c2488eebd85404e0c215c112b46a/model.safetensors)
- [Small](https://huggingface.co/NeoQuasar/Kronos-small/resolve/ac5c409a313c4eadd1ca78201322f5cadb9c34ab/model.safetensors)
- [Base](https://huggingface.co/NeoQuasar/Kronos-base/resolve/31afc02ed0a3cfaa1a97238c8b948d663e45b597/model.safetensors)

如主动选择第三方镜像，可把上述链接中的 https://huggingface.co 换成 https://hf-mirror.com，其余路径和固定版本保持不变。仍须通过相同 SHA256 校验。

例如使用 Base：

~~~text
C:\AI\Kronos\model_files\Kronos-Tokenizer-base\model.safetensors
C:\AI\Kronos\model_files\Kronos-base\model.safetensors
~~~

完整大小与 SHA256 见 [清单](../stock_ui/model_configs/manifest.json)。不要把网页 HTML、Git LFS 指针文本、下载未完成文件或其他版本重命名为权重。

## 权重保存与构造参数错误

新版先校验随包官方配置，再下载权重，从本机配置和真实文件显式加载。

- 配置错误：重新完整解压当前版本，避免混用不同版本的程序和配置。
- 网络错误：检查下载来源和完整报错。
- 文件完整性错误：核对文件大小、SHA256 和固定版本。
- 本机保存错误：查看目标路径、写入权限、磁盘空间以及文件占用信息。
- 初始化错误：如果配置和权重均已校验，保留具体异常供进一步排查。

程序把失效的旧权重保存为 .invalid-时间戳备份，校验成功后才替换。无需清空整个 HF 缓存，也不要关闭 Windows 安全设置来“解决”报错。

## 历史数量不足

5000 根是请求上限。上市时间较晚、无成交日、公开接口覆盖或临时限制可能让实际可用根数更少。页面显示起止日期和实际数量；只要满足所选预测输入长度，可以继续生成预测。

如果连输入窗口都不足，缩短到 128 / 256 根后重试。程序不会补造更早日线。日线保持前复权口径，同一历史不会混接不同来源或未复权段。

## 窗口导致方向变化

128、256、400 根改变的是模型上下文和归一化统计值，均有可能导致不同方向。更长不等于更准确。固定同一股票、数据源、日期、模型和预测期，才能进行可比检查。

如果要评估稳定性，需要另行设计多个历史时点的滚动回测。反复挑选看涨的窗口不能验证预测有效。

## 页面和日期

服务只在本机运行。保持启动窗口打开，使用其实际打开的地址；端口 7070 被占用时启动器会尝试 7071 至 7090。

北京时间 15:15 前使用上一已收盘交易日，周末和节假日使用最近已收盘日。交易日历仅核实至 2026-12-31；超过范围时需要更新，不能把工作日简单当作全部交易日。
