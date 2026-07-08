# 领队协议 (Lead Protocol)

## 角色定义

| 角色 | Agent | 职责 |
|------|-------|------|
| **领队 (Lead)** | 父级 Pi session (我) | 决策轮次、调配资源、判断何时停止 |
| **架构 (Architect)** | `package:planner` | 读规范 → 分析 → 出设计方案 |
| **执行 (Developer)** | `package:worker` | 按设计实现代码 |
| **检测 (Reviewer)** | `package:reviewer` | 审查代码质量和对齐度 |

## 团队模板 (3个)

### 完整轮 (team-round.json)
```
Architect → Developer → Reviewer
```
用于: 新功能、架构变更、复杂重构

### 快速轮 (team-round-fast.json)
```
Developer → Reviewer
```
用于: Bug修复、小调整、UI清理 (无需架构分析)

### 并行轮 (inline graph)
```
Developer-A + Developer-B → Reviewer
```
用于: 两个独立模块同时开发，事后合流审查

## 领队决策规则

### 选择团队规模

| 任务类型 | 模板 | 理由 |
|----------|------|------|
| 新模块/新算法 | 完整轮 | 需要先设计再实现 |
| 参数调整/Bug修复 | 快速轮 | 改动范围小，无需分析 |
| 两个独立文件改动 | 并行轮 | 可同时进行 |
| UI清理/文案修改 | 快速轮 | 纯文本改动 |
| 架构重构 | 完整轮 | 必须规划 |

### 领队决定多少轮

1. **首次**: 使用完整轮确保架构正确
2. **迭代**: 根据 Reviewer 反馈决定是否再来一轮
3. **停止条件**:
   - Reviewer 报告"建议通过" → 停止
   - Reviewer 报告"有条件通过"且 blocker 已修复 → 停止
   - 连续 3 轮未通过 → 领队介入分析根因

### 领队每轮做什么

1. **定义目标** — 明确本轮要完成什么
2. **选择模板** — 完整轮/快速轮/并行轮
3. **编写任务** — 为每个 agent 写清楚 task
4. **启动** — `agent_team start` 或 `graphFile`
5. **监控** — `run_status` 等待完成
6. **检查** — 读 reviewer 输出
7. **决策** — 通过/修补/再来一轮

## 当前项目规范文件

| 文件 | 用途 |
|------|------|
| `../检测.md` | 原始功能规范 (27节) |
| `../改进.md` | 架构升级规范 (计数逻辑、多层检测) |
| `team-round.json` | 完整轮模板 |
| `team-round-fast.json` | 快速轮模板 |

## 启动命令示例

### 完整轮 (新功能开发)
```bash
# 领队手动编辑 team-round.json 的 task 文本, 然后:
agent_team start { graphFile: "team-round.json" }
```

### 快速轮 (Bug修复)
```bash
agent_team start { graphFile: "team-round-fast.json" }
```

### 并行轮 (两个独立修改)
```bash
agent_team start {
  graph: {
    steps: [
      { id: "dev-a", agent: {ref: "package:worker"}, task: "改文件A..." },
      { id: "dev-b", agent: {ref: "package:worker"}, task: "改文件B..." },
      { id: "review", agent: {ref: "package:reviewer"}, task: "审查A和B", needs: ["dev-a", "dev-b"] }
    ]
  }
}
```
