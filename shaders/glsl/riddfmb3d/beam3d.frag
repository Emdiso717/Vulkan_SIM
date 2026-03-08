#version 450

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inViewVec;
layout(location = 3) in vec3 inLightVec;

layout(location = 0) out vec4 outFragColor;

void main() {
  vec3 N = normalize(inNormal);
  // 固定一个从“上方偏前”打下来的光方向，避免底面比顶面亮
  vec3 L = normalize(vec3(0.3, 1.0, 0.4));

  // 纯漫反射 + 环境光，去掉高光，效果更“平滑”
  float ndotl = max(dot(N, L), 0.0);

  // 0.3~1.0 的亮度范围：侧面中灰，正对光时接近白
  float shade = 0.3 + 0.7 * ndotl;

  vec3 baseColor = vec3(0.8, 0.8, 0.8);
  vec3 color = baseColor * shade;

  // 根据视线与法线夹角做一点轮廓线强化（接近轮廓的地方稍微变暗）
  vec3 V = normalize(inViewVec);
  float edge = pow(1.0 - abs(dot(N, V)), 2.0); // 轮廓处 edge 接近 1
  color = mix(color, vec3(0.1), 0.4 * edge);

  // 简单 gamma 校正，让梯度更自然
  color = pow(color, vec3(1.0 / 2.2));
  outFragColor = vec4(color, 1.0);
}