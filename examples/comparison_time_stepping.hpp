#pragma once

// Shared fixed-rate time-stepping configuration for the solver-comparison
// samples.  The JSON configuration has four required fields:
//
// {
//   "substepsPerFrame": 20,
//   "youngsModulus": 1.0e6,
//   "poissonRatio": 0.40,
//   "gravity": 9.8
// }

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

// nlohmann::json is already vendored by TinyGLTF in external/tinygltf.
#include "json.hpp"

#if defined(__ANDROID__)
#include "VulkanAndroid.h"
#include <android/asset_manager.h>
#endif

namespace comparison_time_stepping {

inline constexpr float kFrameDeltaT = 1.0f / 60.0f;
inline constexpr char kConfigAssetPath[] = "configs/solver_comparison.json";

struct Configuration {
  uint32_t substepsPerFrame{20};
  float youngsModulus{1.0e6f};
  float poissonRatio{0.40f};
  // Magnitude only. The comparison uses a fixed negative-Y gravity direction.
  float gravityMagnitude{9.8f};
};

inline bool readTextResource(const std::string &path, std::string &contents) {
#if defined(__ANDROID__)
  if (androidApp == nullptr || androidApp->activity == nullptr ||
      androidApp->activity->assetManager == nullptr) {
    return false;
  }

  AAsset *asset = AAssetManager_open(androidApp->activity->assetManager,
                                     path.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    return false;
  }

  const auto length = static_cast<size_t>(AAsset_getLength(asset));
  const auto *data = static_cast<const char *>(AAsset_getBuffer(asset));
  if (data == nullptr) {
    AAsset_close(asset);
    return false;
  }

  contents.assign(data, length);
  AAsset_close(asset);
  return true;
#else
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  contents.assign(std::istreambuf_iterator<char>(file),
                  std::istreambuf_iterator<char>());
  return true;
#endif
}

inline bool getPositiveUint32(const nlohmann::json &value, uint32_t &parsed) {
  uint64_t result = 0;
  if (value.is_number_unsigned()) {
    result = value.get<uint64_t>();
  } else if (value.is_number_integer()) {
    const int64_t signedValue = value.get<int64_t>();
    if (signedValue > 0) {
      result = static_cast<uint64_t>(signedValue);
    }
  }
  if (result == 0 || result > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  parsed = static_cast<uint32_t>(result);
  return true;
}

inline bool loadConfiguration(const std::string &path, Configuration &config) {
  std::string json;
  if (!readTextResource(path, json)) {
    std::cerr << "comparison time stepping: could not read '" << path
              << "'; using built-in comparison parameters\n";
    return false;
  }

  const nlohmann::json document = nlohmann::json::parse(json, nullptr, false);
  if (document.is_discarded() || !document.is_object()) {
    std::cerr << "comparison time stepping: '" << path
              << "' is not a JSON object; using built-in comparison parameters\n";
    return false;
  }

  const auto substepsIt = document.find("substepsPerFrame");
  const auto youngsIt = document.find("youngsModulus");
  const auto poissonIt = document.find("poissonRatio");
  const auto gravityIt = document.find("gravity");
  if (substepsIt == document.end() || youngsIt == document.end() ||
      poissonIt == document.end() || gravityIt == document.end()) {
    std::cerr << "comparison time stepping: '" << path
              << "' must contain substepsPerFrame, youngsModulus, "
                 "poissonRatio, and gravity; using built-in comparison "
                 "parameters\n";
    return false;
  }

  Configuration parsed = config;
  if (!getPositiveUint32(*substepsIt, parsed.substepsPerFrame) ||
      !youngsIt->is_number() || !poissonIt->is_number() ||
      !gravityIt->is_number()) {
    std::cerr << "comparison time stepping: invalid parameter type in '"
              << path << "'; using built-in comparison parameters\n";
    return false;
  }

  parsed.youngsModulus = youngsIt->get<float>();
  parsed.poissonRatio = poissonIt->get<float>();
  parsed.gravityMagnitude = gravityIt->get<float>();
  if (!std::isfinite(parsed.youngsModulus) ||
      parsed.youngsModulus < 1.0e5f || parsed.youngsModulus > 1.0e8f ||
      !std::isfinite(parsed.poissonRatio) || parsed.poissonRatio < 0.30f ||
      parsed.poissonRatio > 0.49f ||
      !std::isfinite(parsed.gravityMagnitude) ||
      parsed.gravityMagnitude < 0.0f || parsed.gravityMagnitude > 50.0f) {
    std::cerr << "comparison time stepping: parameter range error in '"
              << path << "'; expected E=[1e5,1e8], Pr=[0.30,0.49], "
                 "gravity=[0,50]; using built-in comparison parameters\n";
    return false;
  }

  config = parsed;
  std::cout << "comparison time stepping: fixed frame dt=" << kFrameDeltaT
            << ", substepsPerFrame=" << config.substepsPerFrame
            << ", substep dt="
            << kFrameDeltaT / static_cast<float>(config.substepsPerFrame)
            << ", E=" << config.youngsModulus
            << ", Pr=" << config.poissonRatio
            << ", gravity=(0, -" << config.gravityMagnitude << ", 0)\n";
  return true;
}

} // namespace comparison_time_stepping
