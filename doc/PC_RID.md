# PC 端 RID 三维可变形体示例

`examples/riddfmb3d` 是项目的 PC 端 RID（Regularized Incompressible
Dynamics）三维四面体可变形体示例。它使用 Vulkan Compute Shader 在 GPU 上
完成积分和约束求解，再直接将粒子缓冲作为图形顶点缓冲渲染。

示例默认读取 `assets/models/beam3d.vtk`，通过 `X_MIN` 将 X 最小端的一小段固定，其余部分
可以在重力、弹性和阻尼作用下运动。RID 的每个时间子步包含：

1. `rid_begin.comp`：预测粒子位置并清零约束乘子；
2. `rid_solve.comp`：按无共享顶点的颜色组并行求解四面体约束；
3. `rid_end.comp`：更新速度、阻尼和可选地面碰撞。

## 构建与运行

在仓库根目录构建：

```powershell
cmake -S . -B build
cmake --build build --target riddfmb3d --config Release
```

生成的程序为 `bin/riddfmb3d.exe`。运行时可用以下任一方式指定配置文件：

```powershell
.\bin\riddfmb3d.exe --config .\examples\riddfmb3d\config.txt
.\bin\riddfmb3d.exe --config=.\examples\riddfmb3d\config.txt
.\bin\riddfmb3d.exe -cfg .\examples\riddfmb3d\config.txt
```

配置文件路径可以在仓库外，但 `modelPath` 始终相对于 `assets/` 解析。因此默认
模型应写为 `models/beam3d.vtk`，而不是磁盘绝对路径。

需要导出每帧网格时：

```powershell
.\bin\riddfmb3d.exe --config .\my-rid.txt --write-vtk `
  --vtk-output-dir .\output\rid --vtk-frame-limit 300
```

`--vtk-frame-limit` 必须是正整数；省略它会持续导出，直到程序结束。

## `config.txt` 格式

每行写成 `key = value`；空行和 `#` 之后的内容会被忽略。布尔值只接受
`true`/`false` 或 `1`/`0`。未填写的键保持程序默认值。

| 键 | 格式与有效范围 | 说明 |
| --- | --- | --- |
| `modelPath` | 字符串 | `assets/` 下的四面体 VTK 模型，例如 `models/beam3d.vtk`。 |
| `numSolverIterations` | 无符号整数 | 每个子步的 RID 约束迭代次数。`1` 是最快的起点；提高它通常会使材料更接近约束目标。 |
| `deltaTInv` | 浮点数，`> 0` | 子步长倒数，实际子步长为 `1 / deltaTInv`。在 60 FPS 下，`1200` 对应每帧约 20 个子步。 |
| `density` | 浮点数，`> 0` | 质量密度。 |
| `damping` | 浮点数，`>= 0` | 速度阻尼率；越大，振动衰减越快。 |
| `gravity` | 四个浮点数 `x y z w` | 重力向量。通常写作 `0 -9.8 0 0`。 |
| `youngsModulus` | 浮点数，`> 0` | 杨氏模量；越大越硬。 |
| `poissonRatio` | 浮点数，`(-1, 0.5)` | 泊松比。实际材料通常使用 `0.0` 到 `0.49`。 |
| `fixedSelector` | `X_MIN` 或 `Y_MAX` | 固定端选择方式，见下节。 |
| `fixedRelativeThickness` | 浮点数，`(0, 1]` | 所选端部占模型对应轴长度的比例。 |
| `groundEnabled` | 布尔值 | 是否启用水平地面碰撞。 |
| `groundHeight` | 浮点数 | 地面 Y 坐标。 |
| `groundRestitution` | 浮点数，`>= 0` | 碰撞反弹系数；通常使用 `0` 到 `1`。 |

### 固定端选择

`X_MIN` 固定模型 X 最小端的一段。对 `beam3d.vtk`，`0.08` 表示固定 X 方向
最左侧 8% 的区域：

```text
fixedSelector = X_MIN
fixedRelativeThickness = 0.08
```

`Y_MAX` 用于固定模型最上方的一层，厚度由 `fixedRelativeThickness` 控制：

```text
fixedSelector = Y_MAX
fixedRelativeThickness = 0.08
```

当前程序只接受 `X_MIN` 和 `Y_MAX`。不再使用平面法线、偏移或容差配置。

## 可直接使用的配置示例

以下内容分别保存为独立的 `.txt` 文件后即可传给 `--config`。

### 1. 固定一端的默认梁

```text
modelPath = models/beam3d.vtk
numSolverIterations = 1
deltaTInv = 1200
density = 1000
damping = 1.5
gravity = 0 -9.8 0 0
youngsModulus = 1000000
poissonRatio = 0.40

fixedSelector = X_MIN
fixedRelativeThickness = 0.08

groundEnabled = false
groundHeight = -1
groundRestitution = 0.3
```

### 2. 更软、阻尼更强的悬臂梁

```text
modelPath = models/beam3d.vtk
numSolverIterations = 4
deltaTInv = 1200
density = 1000
damping = 4.0
gravity = 0 -9.8 0 0
youngsModulus = 200000
poissonRatio = 0.35

fixedSelector = X_MIN
fixedRelativeThickness = 0.08

groundEnabled = false
```

### 3. 固定兔子上端

```text
modelPath = models/bunny_3828_asc.vtk
numSolverIterations = 3
deltaTInv = 1800
density = 1000
damping = 1.0
gravity = 0 -9.8 0 0
youngsModulus = 1000000
poissonRatio = 0.40

fixedSelector = Y_MAX
fixedRelativeThickness = 0.08

groundEnabled = false
```

## 调参建议

- 仿真不稳定时，优先提高 `deltaTInv`（更小子步）或提高
  `numSolverIterations`。
- 材料过软时提高 `youngsModulus`；振动过久时提高 `damping`。
- 固定区域不对时，检查 `fixedSelector` 与 `fixedRelativeThickness`：`X_MIN`
  固定 X 最小端，`Y_MAX` 固定 Y 最大端。
- 建议为每次实验创建新的配置文件，不要改动现有脚本依赖的
  `examples/riddfmb3d/config.txt`。
