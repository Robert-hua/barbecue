# Barbecue API 文档

本文档提供了 Barbecue 控制系统的 API 接口说明。该 API 允许用户控制机器人执行各种任务，如抓取、传输和放置物体。

## 基本信息

- **基础 URL**: `http://<host>:<port>`
- **默认端口**: 8888
- **内容类型**: `application/json`

## API 端点

### 1. 获取可用策略列表

获取系统中所有可用的策略（政策）列表。

**请求**:

```
GET /policies
```

**响应**:

```json
{
  "available_policies": ["pick", "transfer", "place"],
  "default_policy": "pick"
}
```

**响应参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| available_policies | array | 可用策略列表 |
| default_policy | string | 默认策略名称（如果已配置） |

### 2. 创建推理任务

创建一个新的机器人控制任务。

**请求**:

```
POST /inference
```

**请求体**:

```json
{
  "task_name": "pick",
  "control_time_s": 15.0,
  "single_task": null
}
```

**请求参数**:

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_name | string | 是 | 任务名称，必须是可用策略之一 ("pick", "transfer", "place") |
| control_time_s | number | 否 | 控制时间（秒），如果未提供则根据任务类型使用不同的默认值：pick (15秒)，transfer (20秒)，place (10秒) |
| single_task | string | 否 | 单一任务标识符（如果适用） |

**响应**:

```json
{
  "task_name": "pick",
  "status": "pending",
  "progress": 0,
  "result": null,
  "error": null
}
```

**响应参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| task_name | string | 任务名称 |
| status | string | 任务状态 ("pending", "running", "completed", "failed") |
| progress | number | 任务进度 (0-100) |
| result | object | 任务结果（仅当任务完成时） |
| error | string | 错误信息（仅当任务失败时） |

### 3. 获取任务状态

获取当前/最新任务的状态。

**请求**:

```
GET /inference/status
```

**响应**:

```json
{
  "task_name": "pick",
  "status": "running",
  "progress": 50,
  "result": null,
  "error": null
}
```

**响应参数**:

| 参数 | 类型 | 描述 |
|------|------|------|
| task_name | string | 任务名称 |
| status | string | 任务状态 ("pending", "running", "completed", "failed") |
| progress | number | 任务进度 (0-100) |
| result | object | 任务结果（仅当任务完成时） |
| error | string | 错误信息（仅当任务失败时） |

## 状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误（例如，请求了不可用的策略） |
| 500 | 服务器内部错误 |

## 任务结果示例

当任务成功完成时，`result` 字段将包含以下信息：

```json
{
  "execution_time": 28.5,
  "control_time_s": 15.0,
  "single_task": null
}
```

| 参数 | 类型 | 描述 |
|------|------|------|
| execution_time | number | 任务执行实际耗时（秒） |
| control_time_s | number | 任务控制时间设置（秒） |
| single_task | string | 单一任务标识符（如果适用） |

## 配置说明

服务器配置可以通过 `config.json` 文件提供，或者在启动时通过命令行参数指定：

```bash
python server.py --config config.json --host 0.0.0.0 --port 8888
```

### 配置文件格式

```json
{
  "robot_type": "so100",
  "policies": {
    "pick": {
      "path": "/path/to/pick_policy",
      "metadata_path": "/path/to/metadata",
      "device": "cuda",
      "use_amp": true,
      "type": "act",
      "fps": 30,
      "control_time_s": 30.0
    },
    "transfer": {
      "path": "/path/to/transfer_policy",
      "metadata_path": "/path/to/metadata"
    },
    "place": {
      "path": "/path/to/place_policy",
      "metadata_path": "/path/to/metadata"
    }
  },
  "default_task": "pick",
  "control_time_s": 15.0,
  "display_cameras": false
}
```

## 使用示例

### 使用 curl 获取可用策略

```bash
curl -X GET http://localhost:8888/policies
```

### 使用 curl 创建推理任务

```bash
curl -X POST http://localhost:8888/inference \
  -H "Content-Type: application/json" \
  -d '{"task_name": "pick", "control_time_s": 30.0}'
```

### 使用 curl 获取任务状态

```bash
curl -X GET http://localhost:8888/inference/status
```

## 错误处理

当出现错误时，API 将返回适当的 HTTP 状态码和错误详情：

```json
{
  "detail": "Policy pick not available"
}
```
