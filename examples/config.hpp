#pragma once

// Shared runtime config helper for examples.
// It reads simple txt files made of "key = value" lines, ignores empty lines
// and '#' comments, and helps examples fetch a config path from command-line
// args such as "--config path", "--config=path", "-cfg path", or a plain
// "*.txt" argument. Example-specific configuration structs and loaders can
// live here too, so the example source files stay focused on simulation code.

#include "glm/glm.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace example_config {

struct Entry {
  std::string value;
  uint32_t line{0};
};

using Entries = std::unordered_map<std::string, Entry>;

inline std::string trim(std::string value) {
  const auto isNotSpace = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), isNotSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(),
              value.end());
  return value;
}

inline std::string
getConfigPathFromArgs(const std::vector<const char *> &args) {
  for (size_t i = 1; i < args.size(); i++) {
    const std::string arg = args[i];
    const std::string prefix = "--config=";
    if (arg.rfind(prefix, 0) == 0) {
      return arg.substr(prefix.size());
    }
    if ((arg == "--config" || arg == "-config" || arg == "-cfg") &&
        i + 1 < args.size()) {
      return args[i + 1];
    }
    if (arg.size() >= 4 && arg.substr(arg.size() - 4) == ".txt") {
      return arg;
    }
  }
  return "";
}

inline bool hasCommandLineFlag(const std::vector<const char *> &args,
                               const std::string &flag) {
  return std::any_of(args.begin(), args.end(), [&flag](const char *arg) {
    return arg != nullptr && flag == arg;
  });
}

inline bool parseFloat(const std::string &value, float &out) {
  char *end = nullptr;
  const float parsed = std::strtof(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0') {
    return false;
  }
  out = parsed;
  return true;
}

inline bool parseUint32(const std::string &value, uint32_t &out) {
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(parsed);
  return true;
}

inline bool parseBool(const std::string &value, bool &out) {
  if (value == "true" || value == "1") {
    out = true;
    return true;
  }
  if (value == "false" || value == "0") {
    out = false;
    return true;
  }
  return false;
}

inline bool parseVec4(const std::string &value, glm::vec4 &out) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');

  std::istringstream stream(normalized);
  glm::vec4 parsed{};
  if (!(stream >> parsed.x >> parsed.y >> parsed.z >> parsed.w)) {
    return false;
  }

  std::string trailing;
  if (stream >> trailing) {
    return false;
  }

  out = parsed;
  return true;
}

inline bool loadTxt(const std::string &path, Entries &entries,
                    const std::string &logPrefix = "config") {
  entries.clear();

  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << logPrefix << ": failed to open config file '" << path << "'\n";
    return false;
  }

  std::string line;
  uint32_t lineNumber = 0;
  while (std::getline(file, line)) {
    lineNumber++;
    const size_t commentPos = line.find('#');
    if (commentPos != std::string::npos) {
      line.erase(commentPos);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    size_t separator = line.find('=');
    if (separator == std::string::npos) {
      separator = line.find_first_of(" \t");
    }
    if (separator == std::string::npos) {
      std::cerr << logPrefix << ": invalid config line " << lineNumber << ": "
                << line << "\n";
      continue;
    }

    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    if (key.empty() || value.empty()) {
      std::cerr << logPrefix << ": invalid config line " << lineNumber << ": "
                << line << "\n";
      continue;
    }

    entries[key] = {value, lineNumber};
  }

  return true;
}

struct Riddfmb3dConfiguration {
  std::string modelPath = "models/bunny_small(1).vtk";
  uint32_t numSolverIterations{1};
  float deltaTInv{300.0f};
  float density{1000.0f};
  glm::vec4 gravity{0.0f, -9.8f, 0.0f, 0.0f};
  float youngsModulus{1000000.0f};
  float poissonRatio{0.49f};
  bool fixedPlaneEnabled{false};
  glm::vec4 fixedPlaneNormal{1.0f, 0.0f, 0.0f, 0.0f};
  float fixedPlaneOffset{0.0f};
  float fixedPlaneTolerance{1e-4f};
  bool groundEnabled{true};
  float groundHeight{-1.0f};
  float groundRestitution{0.3f};
};

inline glm::vec4 lameParameters(const Riddfmb3dConfiguration &config) {
  const float lambda = config.youngsModulus * config.poissonRatio /
                       ((1.0f + config.poissonRatio) *
                        (1.0f - 2.0f * config.poissonRatio));
  const float mu = config.youngsModulus / (2.0f * (1.0f + config.poissonRatio));
  return glm::vec4(lambda, mu, 0.0f, 0.0f);
}

inline void loadRiddfmb3dConfiguration(const std::vector<const char *> &args,
                                       Riddfmb3dConfiguration &config) {
  const std::string configPath = getConfigPathFromArgs(args);
  if (configPath.empty()) {
    std::cout << "riddfmb3d: no config file specified, using defaults\n";
    return;
  }

  Entries entries;
  if (!loadTxt(configPath, entries, "riddfmb3d")) {
    std::cerr << "riddfmb3d: using defaults\n";
    return;
  }

  for (const auto &[key, entry] : entries) {
    const std::string &value = entry.value;
    if (key == "modelPath") {
      config.modelPath = value;
    } else if (key == "numSolverIterations") {
      uint32_t parsed = 0;
      if (parseUint32(value, parsed)) {
        config.numSolverIterations = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid numSolverIterations at line "
                  << entry.line << "\n";
      }
    } else if (key == "deltaTInv") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed > 0.0f) {
        config.deltaTInv = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid deltaTInv at line " << entry.line
                  << "\n";
      }
    } else if (key == "density") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed > 0.0f) {
        config.density = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid density at line " << entry.line
                  << "\n";
      }
    } else if (key == "gravity") {
      glm::vec4 parsed{};
      if (parseVec4(value, parsed)) {
        config.gravity = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid gravity at line " << entry.line
                  << "\n";
      }
    } else if (key == "youngsModulus") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed > 0.0f) {
        config.youngsModulus = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid youngsModulus at line "
                  << entry.line << "\n";
      }
    } else if (key == "poissonRatio") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed > -1.0f && parsed < 0.5f) {
        config.poissonRatio = parsed;
      } else {
        std::cerr << "riddfmb3d: poissonRatio must be in (-1, 0.5) at line "
                  << entry.line << "\n";
      }
    } else if (key == "fixedPlaneEnabled") {
      bool parsed = false;
      if (parseBool(value, parsed)) {
        config.fixedPlaneEnabled = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid fixedPlaneEnabled at line "
                  << entry.line << "\n";
      }
    } else if (key == "fixedPlaneNormal") {
      glm::vec4 parsed{};
      if (parseVec4(value, parsed)) {
        config.fixedPlaneNormal = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid fixedPlaneNormal at line "
                  << entry.line << "\n";
      }
    } else if (key == "fixedPlaneOffset") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed)) {
        config.fixedPlaneOffset = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid fixedPlaneOffset at line "
                  << entry.line << "\n";
      }
    } else if (key == "fixedPlaneTolerance") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed >= 0.0f) {
        config.fixedPlaneTolerance = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid fixedPlaneTolerance at line "
                  << entry.line << "\n";
      }
    } else if (key == "groundEnabled") {
      bool parsed = false;
      if (parseBool(value, parsed)) {
        config.groundEnabled = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid groundEnabled at line " << entry.line
                  << "\n";
      }
    } else if (key == "groundHeight") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed)) {
        config.groundHeight = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid groundHeight at line " << entry.line
                  << "\n";
      }
    } else if (key == "groundRestitution") {
      float parsed = 0.0f;
      if (parseFloat(value, parsed) && parsed >= 0.0f) {
        config.groundRestitution = parsed;
      } else {
        std::cerr << "riddfmb3d: invalid groundRestitution at line "
                  << entry.line << "\n";
      }
    } else {
      std::cerr << "riddfmb3d: unknown config key '" << key << "' at line "
                << entry.line << "\n";
    }
  }

  const glm::vec4 lame = lameParameters(config);
  std::cout << "riddfmb3d config: modelPath=" << config.modelPath
            << ", numSolverIterations=" << config.numSolverIterations
            << ", deltaTInv=" << config.deltaTInv
            << ", deltaT=" << 1.0f / config.deltaTInv
            << ", density=" << config.density
            << ", gravity=(" << config.gravity.x << ", " << config.gravity.y
            << ", " << config.gravity.z << ", " << config.gravity.w << ")"
            << ", youngsModulus=" << config.youngsModulus
            << ", poissonRatio=" << config.poissonRatio
            << ", lame=(" << lame.x << ", " << lame.y << ", " << lame.z
            << ", " << lame.w << ")"
            << ", fixedPlaneEnabled=" << config.fixedPlaneEnabled
            << ", fixedPlaneNormal=(" << config.fixedPlaneNormal.x << ", "
            << config.fixedPlaneNormal.y << ", " << config.fixedPlaneNormal.z
            << "), fixedPlaneOffset=" << config.fixedPlaneOffset
            << ", fixedPlaneTolerance=" << config.fixedPlaneTolerance
            << ", groundEnabled=" << config.groundEnabled
            << ", groundHeight=" << config.groundHeight
            << ", groundRestitution=" << config.groundRestitution << "\n";
}

} // namespace example_config
