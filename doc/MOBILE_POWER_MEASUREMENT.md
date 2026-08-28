# Android 功耗测量：独立 XPBD 与 RID 应用

为避免展示 demo 中可切换的双后端影响手机功耗测量，项目提供了两个独立、锁定后端的
应用。`xpbdvsrid3d` 仍保持为展示和交互对比 demo，不用于功耗测量。

| 测量应用 | PC target | Android module / 包名 | 锁定后端 |
| --- | --- | --- | --- |
| XPBD | `xpbdjacobi3d` | `xpbdjacobi3d` / `de.saschawillems.xpbdjacobi3d` | Jacobi XPBD（距离 + 体积约束） |
| RID | `ridpower3d` | `ridpower3d` / `de.saschawillems.ridpower3d` | RID |

两个应用复用相同的模型、场景、物理参数、渲染流程和 60 Hz 外层时间步设置。它们的
差异只在求解器后端，因此适合进行对照测量。

## 独立性

- XPBD 应用固定选择 Jacobi XPBD，不提供 Solver 切换控件；RID 缓冲、管线和 compute
  dispatch 不会在运行时初始化或提交。
- RID 应用固定选择 RID，不提供 Solver 切换控件；Jacobi XPBD 缓冲、管线和 compute
  dispatch 不会在运行时初始化或提交。
- RID APK 只打包 RID compute shader；仍会携带共用的图形 shader 和扭转边界 shader，
  它们不属于 XPBD 求解。
- 两个应用使用不同包名，可以同时安装到同一手机，便于轮流测量。

## 保持测量条件一致

两个应用均从 `assets/configs/solver_comparison.json` 读取以下配置：

```json
{
  "substepsPerFrame": 30,
  "youngsModulus": 1.0e7,
  "poissonRatio": 0.40,
  "gravity": 0.0,
  "scene": "cantilever",
  "solver": "rid"
}
```

测量应用会忽略 `solver` 字段，并选择各自锁定的后端；保留该字段是为了继续兼容
`xpbdvsrid3d` demo。测试 XPBD 与 RID 时应保持其余字段完全一致，尤其是：

- `substepsPerFrame`
- `youngsModulus` 和 `poissonRatio`
- `gravity`
- `scene`

支持的场景为 `cantilever`、`beam_twist_3pi` 和 `bunny_squash`。修改 JSON 后必须重新
构建并安装两个 APK，确保设备上的 assets 一致。

## Android：推荐 Android Studio

推荐使用 Android Studio 打开仓库中的 `android/` 目录。选择以下 module 之一并点击
**Run**，Android Studio 会自动完成 CMake/Gradle 构建、安装和推送：

- `xpbdjacobi3d`：测量 XPBD
- `ridpower3d`：测量 RID

进行对比时，先完全停止一个应用，再启动另一个；在相同电量、温度、屏幕亮度、设备
性能模式和 JSON 配置下分别采集功耗。

## Android：命令行构建与安装

在 `android/` 目录分别构建：

```powershell
.\gradlew.bat :xpbdjacobi3d:assembleDebug
.\gradlew.bat :ridpower3d:assembleDebug
```

APK 由项目输出脚本写入 `android/examples/bin/`：

```powershell
adb install -r .\examples\bin\xpbdjacobi3d-debug.apk
adb install -r .\examples\bin\ridpower3d-debug.apk
```

启动命令：

```powershell
adb shell monkey -p de.saschawillems.xpbdjacobi3d 1
adb shell monkey -p de.saschawillems.ridpower3d 1
```

设备需支持 Vulkan，当前 Android 构建默认 ABI 为 `arm64-v8a`。

## PC 端验证

独立 PC 目标可用于确认相同配置下的功能行为：

```powershell
cmake --build build --target xpbdjacobi3d ridpower3d --config Release
.\bin\xpbdjacobi3d.exe
.\bin\ridpower3d.exe
```

PC 可执行文件与 Android 应用一样锁定后端；它们不是 `xpbdvsrid3d` 的替代展示程序。
