#version 450

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inViewVec;

layout(location = 0) out vec4 outFragColor;

void main() {
  vec3 normal = normalize(inNormal);
  vec3 light = normalize(vec3(0.3, 1.0, 0.4));
  float diffuse = max(dot(normal, light), 0.0);
  vec3 color = vec3(0.20, 0.60, 0.95) * (0.30 + 0.70 * diffuse);
  vec3 view = normalize(inViewVec);
  float edge = pow(1.0 - abs(dot(normal, view)), 2.0);
  color = mix(color, vec3(0.06), 0.35 * edge);
  outFragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
