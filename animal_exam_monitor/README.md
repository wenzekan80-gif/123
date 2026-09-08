# 湖北省实验动物考试监控

监控湖北省实验动物公共服务平台的“能力提升/评价”通知。一旦出现此前未见的新通知，GitHub Actions 会自动提醒。

## 默认提醒方式

1. 自动创建 GitHub Issue，并在正文中 @wenzekan80-gif。
2. 可选：配置 PushPlus 后同步推送到微信。

## 监控频率

GitHub Actions 每 5 分钟运行一次：

```yaml
- cron: "*/5 * * * *"
```

注意：GitHub 的 schedule 不是严格实时任务，高峰期可能延迟几分钟。因此这是“近实时”监控，而不是毫秒级监控。

## 首次运行

第一次正常运行只建立当前公告基线，不会把历史通知全部当作新通知发送。

之后只要官方页面出现新的 `portal/exam/details?id=...` 通知，就会触发提醒。

## 微信提醒（推荐）

如果需要手机上更醒目的通知，可以在仓库：

`Settings -> Secrets and variables -> Actions -> New repository secret`

新增：

- Name: `PUSHPLUS_TOKEN`
- Value: 你的 PushPlus token

不配置也不影响 GitHub Issue 提醒。

## 手动测试

进入仓库：

`Actions -> 湖北实验动物考试监控 -> Run workflow`

勾选 `Send a test notification only`，即可测试提醒链路。

## 防漏机制

- 优先直接解析官方通知列表。
- 如果官网内容由 JavaScript 动态加载，自动回退到 Selenium + Chrome 渲染。
- 页面结构改变导致抓取不到公告时，任务会直接失败，而不是静默认为“没有新通知”。
- `state.json` 记录已见通知 ID，防止同一条通知每 5 分钟重复提醒。

## 重点识别

脚本会把含有以下关键词、且不是专项培训的通知标为 `HIGH`：

- 实验动物从业人员
- 动物实验从业人员
- 从业人员技术咨询
- 能力辅导评价
- 能力提升及评价
- 能力评价

即使标题措辞变化，只要官方能力提升页面出现新的通知 ID，仍会被记录和提醒，避免因为关键词变化漏报。
