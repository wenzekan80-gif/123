# 湖北省实验动物考试监控

监控湖北省实验动物公共服务平台的“能力提升/评价”通知。一旦出现此前未见的新通知，GitHub Actions 会自动提醒。

## 提醒方式

默认：

1. 自动创建 GitHub Issue，并在正文中 @wenzekan80-gif。

可选增强：

2. 配置 SMTP 后直接发邮件到你指定的邮箱（支持 QQ 邮箱）。
3. 配置 PushPlus 后同步推送到微信。

> 不要把手机号、邮箱密码、QQ 邮箱授权码直接写进代码或公开仓库。全部使用 GitHub Actions Secrets。

## 监控频率：近似每分钟

GitHub 官方 `schedule` 的最短周期是 5 分钟，因此不能把 cron 直接写成真正的 1 分钟定时。

本项目采用折中方案：

- GitHub 每 5 分钟启动一次任务；
- 每次任务内部连续检查 5 次；
- 两次检查之间 `sleep 60` 秒。

因此在 GitHub runner 正常启动的情况下，检测频率接近每分钟一次。

注意：GitHub 定时任务本身可能因为平台负载而延迟，所以这不是严格的 60 秒 SLA。若需要真正稳定的 1 分钟甚至更快监控，应改用常驻服务器 / Cloudflare Worker / VPS。

## QQ 邮箱直接提醒

仓库路径：

`Settings -> Secrets and variables -> Actions -> New repository secret`

至少新增 3 个 Secret：

- `SMTP_USER`：用于发信的 QQ 邮箱账号，例如 `xxxxxxxxxx@qq.com`
- `SMTP_AUTH_CODE`：QQ 邮箱 SMTP 授权码（不是 QQ 密码）
- `ALERT_EMAIL`：接收报警的邮箱地址

QQ 邮箱默认已经在代码中使用：

- Host: `smtp.qq.com`
- Port: `465`
- SSL: 开启

所以通常不需要另外设置。如果以后换其他邮件服务，可增加：

- `SMTP_HOST`
- `SMTP_PORT`

### QQ 邮箱授权码

在 QQ 邮箱网页/设置中开启 SMTP 服务并生成授权码，然后把授权码只放入 GitHub Secret `SMTP_AUTH_CODE`。

**不要把授权码发到聊天、Issue、README 或代码里。**

## 微信提醒

新增 Secret：

- Name: `PUSHPLUS_TOKEN`
- Value: 你的 PushPlus token

不配置也不影响 GitHub Issue 和邮件提醒。

## 手动测试通知

工作流合并到默认分支后，进入：

`Actions -> 湖北实验动物考试监控 -> Run workflow`

勾选：

`Send a test notification only`

然后运行。它会测试所有已经配置的通知渠道：

- GitHub Issue
- SMTP 邮件
- PushPlus 微信

## 首次正常运行

第一次正常监控只建立当前公告基线，不会把历史通知全部当作新通知发送。

之后只要官方页面出现新的 `portal/exam/details?id=...` 通知，就会触发提醒。

## 防漏机制

- 优先直接解析官方通知列表。
- 如果官网内容由 JavaScript 动态加载，自动回退到 Selenium + Chrome 渲染。
- 页面结构改变导致抓取不到公告时，任务会直接失败，而不是静默认为“没有新通知”。
- `state.json` 记录已见通知 ID，防止同一条通知重复提醒。
- 单个通知渠道失败不会阻止其他渠道继续报警，也不会因为状态未保存而无限重复发信。

## 重点识别

脚本会把含有以下关键词、且不是专项培训的通知标为 `HIGH`：

- 实验动物从业人员
- 动物实验从业人员
- 从业人员技术咨询
- 能力辅导评价
- 能力提升及评价
- 能力评价

即使标题措辞变化，只要官方能力提升页面出现新的通知 ID，仍会被记录和提醒，避免因为关键词变化漏报。
