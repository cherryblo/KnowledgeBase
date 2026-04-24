# GitHub Workflow 测试结果分析工具

## 功能介绍

这是一个用于分析 GitHub 工作流中测试用例执行结果的工具。它可以：

- 获取仓库的工作流运行记录
- 分析每个工作流运行的作业状态
- 提取失败作业的测试日志
- 统计测试用例的执行结果（通过、失败、跳过）
- 显示详细的测试失败信息
- 生成 MD 格式的测试报告
- 支持从测试报告创建 GitHub issue
- 处理空日志文件的情况

## 环境要求

- Node.js 14+
- GitHub 个人访问令牌（具有 `repo` 权限）

## 安装步骤

1. 克隆或下载本项目到本地
2. 安装依赖：
   ```bash
   npm install
   ```

## 配置步骤

1. 复制 `.env.example` 文件为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，填写以下信息：
   - `GITHUB_TOKEN`：你的 GitHub 个人访问令牌
   - `GITHUB_OWNER`：仓库所有者（用户名或组织名）
   - `GITHUB_REPO`：仓库名称
   - `GITHUB_BRANCH`：分支名称（可选，如果指定，只分析该分支下的工作流任务）

## 使用方法

### 分析所有工作流

```bash
node analyze-workflow-tests.js
```

### 分析指定工作流

```bash
# 直接指定工作流名称
node analyze-workflow-tests.js "CI"

# 使用 --workflow 参数指定
node analyze-workflow-tests.js --workflow "CI"
```

### 分析指定工作流运行

```bash
# 使用 --run 参数指定工作流运行ID
node analyze-workflow-tests.js --run 1234567890

# 同时指定工作流名称和运行ID
node analyze-workflow-tests.js --workflow "CI" --run 1234567890
```

### 从测试报告创建 GitHub issue

当你确认测试报告内容正确后，可以使用以下命令从报告创建 GitHub issue：

```bash
# 使用 --create-issue 参数指定报告文件路径
node analyze-workflow-tests.js --create-issue ./reports/test_report_workflow_1234567890_2026-04-06T09-37-25-056Z.md
```

**命令说明**：
- `--create-issue`：指定要从测试报告创建 GitHub issue
- 后面的参数是测试报告文件的路径，这个路径会在分析完成后显示在输出中

**创建过程**：
1. 工具会读取指定的 MD 报告文件
2. 从报告中提取工作流名称、创建时间和执行结论
3. 生成英文格式的 issue 标题：`[TestReport] ${workflowName} was executed at ${createdTime}, result: ${conclusion}`
4. 将完整的 MD 报告内容作为 issue 正文
5. 自动添加 `bug` 标签
6. 返回创建的 issue URL

**注意事项**：
- 确保报告文件存在且格式正确
- 需要在 `.env` 文件中正确配置 GitHub 令牌和仓库信息
- GitHub API 有速率限制，请合理使用

## 输出示例

### 分析输出

```
分析 owner/repo 的工作流测试结果...
找到 5 个工作流运行记录

=== 工作流: CI (ID: 123456789) ===
状态: completed
结论: failure
触发事件: push
创建时间: 2026-04-03 10:00:00
作业数量: 2

  - 作业: Test (ID: 987654321)
    状态: completed
    结论: failure
    📋 Analyzing test job logs...
    Logs saved to: ./logs/job_987654321_logs.txt
    Test Results:
      Total: 10
      Passed: 8
      Failed: 2
      Skipped: 0
    Failed Tests:
      1. 2026-04-03T10:05:00.0000000Z   test/example.test.js (exit code 1)
      2. 2026-04-03T10:05:05.0000000Z   test/another.test.js (exit code 1)
    Passed Tests:
      1. 2026-04-03T10:04:30.0000000Z   test/passing.test.js
      2. 2026-04-03T10:04:35.0000000Z   test/another-passing.test.js
      ... and 6 more passed tests

  - 作业: Build (ID: 112233445)
    状态: completed
    结论: success

MD report generated: ./reports/test_report_workflow_123456789_2026-04-06T09-37-25-056Z.md

To create a GitHub issue with this report, run:
node analyze-workflow-tests.js --create-issue ./reports/test_report_workflow_123456789_2026-04-06T09-37-25-056Z.md
```

### MD 报告示例

生成的 MD 报告包含以下内容：

- 执行信息（工作流ID、名称、状态、结论等）
- Job 执行统计结果表格（包含序号、Job 名称、状态、结论、测试数量等）
- 失败用例详情表格（包含序号、测试文件名称、所属 Job、分析责任人等）
- 汇总统计信息

## 注意事项

- 测试结果解析逻辑基于简单的字符串匹配，可能需要根据实际的测试框架输出格式进行调整
- 工具默认只获取最新的一个工作流运行，以提高分析速度并减少API请求次数
- 当指定工作流名称时，工具会尝试获取更多工作流运行来找到匹配的
- GitHub API 有速率限制，请合理使用

## 扩展建议

- 添加测试结果导出功能（CSV、JSON 等格式）
- 实现历史趋势分析，比较不同时间段的测试结果
- 支持更多测试框架的日志解析
- 添加邮件或 Slack 通知功能