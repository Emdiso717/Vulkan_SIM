# Vulkan 三维可变形体仿真

本项目基于 **Vulkan** 实现三维四面体可变形体的 GPU 实时仿真与渲染，重点比较两类并行约束求解方法：Jacobi XPBD（Extended Position-Based Dynamics）与 RID（Regularized Incompressible Dynamics）。项目可在 PC 上运行，并提供 Android 构建目标，适用于求解器行为、性能及移动端功耗的对照实验。

## 已实现内容

- **XPBD 与 RID 对比示例**（`xpbdvsrid3d`）：在同一应用中按需切换 Jacobi XPBD 和 RID 后端；两者复用模型、渲染流程和时间步设置，便于比较稳定性、形变与性能。
- **独立求解器测量程序**：`xpbdjacobi3d` 固定使用 Jacobi XPBD，`ridpower3d` 固定使用 RID，避免运行时切换逻辑干扰移动端功耗测量。
- **独立 PC RID 示例**（`riddfmb3d`）：支持从配置文件读取材料、时间步、固定端、重力和地面碰撞等参数，并可导出逐帧 VTK 网格。
- **三类测试场景**：悬臂梁、扭转梁和兔子压缩恢复；模型与公共配置位于 `assets/`。
- **GPU 仿真和渲染**：使用 Vulkan Compute Shader 完成积分与约束求解，粒子缓冲直接作为顶点缓冲渲染；提供 ImGui 参数面板。

## 技术框架

项目采用 C++20 和 CMake 构建，以 Vulkan 为图形与计算 API。代码结构在 Sascha Willems 的 Vulkan 示例框架基础上扩展：

- `base/`：Vulkan 设备、交换链、缓冲、纹理、相机、UI 等通用封装；
- `examples/`：各仿真应用及其 CMake 目标；
- `shaders/glsl/`：GLSL 图形/计算着色器及编译后的 SPIR-V；
- `assets/`：模型、纹理、字体与运行配置；
- `external/`：随仓库提供的依赖，包括 ImGui、GLM、Eigen、tinygltf 与 KTX；
- `android/`：Android Gradle 工程与应用模块。

## 构建与运行（PC）

### 获取代码

克隆仓库后，请初始化并同步所有子模块：

```powershell
git submodule update --init --recursive
```

### 前置条件

- 支持 C++20 的编译器（Windows 推荐 Visual Studio 2022）；
- CMake 3.10 或更高版本；
- Vulkan SDK，以及支持 Vulkan 的显卡驱动。

在仓库根目录配置并构建所需目标：

```powershell
cmake -S . -B build
cmake --build build --target xpbdvsrid3d --config Release
```

生成的可执行文件位于 `bin/`。例如运行求解器对比程序：

```powershell
.\bin\xpbdvsrid3d.exe
```

可替换构建目标为：

| 目标 | 用途 |
| --- | --- |
| `xpbdvsrid3d` | XPBD 与 RID 的可切换对比程序 |
| `xpbdjacobi3d` | 固定 Jacobi XPBD 的测量程序 |
| `ridpower3d` | 固定 RID 的测量程序 |
| `riddfmb3d` | 可通过文本配置的 PC RID 示例 |

例如一次构建两个固定后端程序：

```powershell
cmake --build build --target xpbdjacobi3d ridpower3d --config Release
```

## 详细文档

本 README 仅提供项目总览；示例配置、运行参数、Android 构建与功耗测量请查阅 [`doc/`](doc/) 目录下的文档：

- [`doc/XPBD_VS_RID_3D.md`](doc/XPBD_VS_RID_3D.md)：XPBD/RID 对比示例、配置及 Android 部署；
- [`doc/PC_RID.md`](doc/PC_RID.md)：PC RID 示例的配置格式、运行和 VTK 导出；
- [`doc/MOBILE_POWER_MEASUREMENT.md`](doc/MOBILE_POWER_MEASUREMENT.md)：Android 独立应用的构建、安装与功耗对照测量。
