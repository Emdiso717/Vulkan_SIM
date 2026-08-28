# XPBD vs RID 三维对比示例

`examples/xpbdvsrid3d` 是一个同时包含 Jacobi XPBD 与 RID 后端的三维四面体
可变形体对比示例。两种求解器共用模型、显示和时间步设置，因此可在同一场景中
切换并比较它们的稳定性、形变和性能。

该示例既可在 **PC** 端运行，也可构建为 **Android APK 并通过 ADB 推送到设备**。
Android 工程只注册这一个应用模块：`xpbdvsrid3d`。

## 求解器与场景

| 项目 | 说明 |
| --- | --- |
| Jacobi XPBD | 距离约束和体积约束分别执行 evaluate/apply 阶段，适合并行的 Jacobi 求解。 |
| RID | 将四面体按无共享顶点的颜色组排列，再执行 RID 约束求解。 |
| `cantilever` | 默认梁模型；X 最小端固定。 |
| `beam_twist_3pi` | 梁两端按相反方向施加扭转。 |
| `bunny_squash` | 以压平的兔子初始状态开始恢复形变。 |

启动时只初始化 JSON 指定的求解器；切换 UI 中的 `Solver` 后，另一个后端会按需
初始化并自动重置粒子状态。

## PC 端构建与运行

在仓库根目录执行：

```powershell
cmake -S . -B build
cmake --build build --target xpbdvsrid3d --config Release
.\bin\xpbdvsrid3d.exe
```

程序启动后可在 ImGui 面板中：

- 用 `Solver` 切换 `Jacobi XPBD (distance + volume)` 与 `RID`；
- 调整杨氏模量、泊松比和重力；调整参数会自动请求重置；
- 使用 `Pause/Resume simulation` 暂停或继续；
- 使用 `Reset simulation` 回到当前场景的初始状态。

## 对比配置

PC 和 Android 都读取同一文件：

```text
assets/configs/solver_comparison.json
```

四个基础字段必须存在；`scene` 和 `solver` 可选。当前仓库配置以 RID 和悬臂梁
启动。

| 字段 | 有效值 | 说明 |
| --- | --- | --- |
| `substepsPerFrame` | 正整数 | 一个固定的 `1/60 s` 渲染步包含的求解子步数。 |
| `youngsModulus` | `1e5` 到 `1e8` | 材料杨氏模量。 |
| `poissonRatio` | `0.30` 到 `0.49` | 材料泊松比。 |
| `gravity` | `0` 到 `50` | 向下的重力大小；实际方向固定为负 Y。 |
| `scene` | `cantilever`、`beam_twist_3pi`、`bunny_squash` | 场景选择。 |
| `solver` | `jacobi`、`rid` | 启动时使用的求解器。 |

### 示例：RID 悬臂梁

```json
{
  "substepsPerFrame": 30,
  "youngsModulus": 1.0e7,
  "poissonRatio": 0.40,
  "gravity": 9.8,
  "scene": "cantilever",
  "solver": "rid"
}
```

### 示例：Jacobi XPBD 扭转梁

```json
{
  "substepsPerFrame": 30,
  "youngsModulus": 1.0e7,
  "poissonRatio": 0.40,
  "gravity": 0.0,
  "scene": "beam_twist_3pi",
  "solver": "jacobi"
}
```

### 示例：RID 兔子恢复

```json
{
  "substepsPerFrame": 40,
  "youngsModulus": 5.0e6,
  "poissonRatio": 0.40,
  "gravity": 0.0,
  "scene": "bunny_squash",
  "solver": "rid"
}
```

修改 JSON 后重新启动程序即可生效。Android 构建时，Gradle 的 `copyTask` 会将该
配置、`beam3d.vtk`、`bunny_3828_asc.vtk` 以及两套求解器的 SPIR-V shader 拷贝到
APK assets 中。

## Android 构建与部署

Android 工程位于 `android/`，包名为
`de.saschawillems.xpbdvsrid3d`，默认构建 ABI 为 `arm64-v8a`。目标设备必须支持
Vulkan。

### 推荐：使用 Android Studio

推荐直接用 Android Studio 打开 `android/` 目录，选择 `xpbdvsrid3d` module 与已连接
设备后点击 **Run**。Android Studio 会自动完成 CMake/Gradle 构建、APK 安装和推送，
是调试与反复推包最快捷的方式。

### 命令行构建与推送

首次使用前，安装 Android SDK、NDK 和 CMake；若 `android/local.properties` 不存在，
创建它并指定本机 SDK 路径，例如：

```properties
sdk.dir=C:\\Android\\Sdk
```

在 `android/` 目录执行 Debug 构建：

```powershell
.\gradlew.bat :xpbdvsrid3d:assembleDebug
```

项目的 APK 输出脚本会将 APK 放在：

```text
android/examples/bin/xpbdvsrid3d-debug.apk
```

连接已启用 USB 调试的 Android 设备后，安装并启动：

```powershell
adb devices
adb install -r .\examples\bin\xpbdvsrid3d-debug.apk
adb shell monkey -p de.saschawillems.xpbdvsrid3d 1
```

横屏启动后，使用触控打开并操作 ImGui 面板以切换求解器和调整参数。

## 常见检查

- APK 中缺少模型或 shader 时，先执行一次 `assembleDebug`，让 `copyTask` 重新生成
  module 的 assets。
- 改动 `solver_comparison.json` 后必须重新打包 Android APK；设备上已安装的 APK 不会
  自动读取 PC 工作区中的新配置。
- 若运行时切换求解器，示例会重置状态；这避免两套求解器使用彼此已经变形的结果作为
  初始状态。
