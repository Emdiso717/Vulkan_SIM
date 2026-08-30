// Power-measurement executable: only the Jacobi XPBD backend is selected.
#include "../comparison_time_stepping.hpp"
#include "VulkanglTFModel.h"
#include "glm/gtc/matrix_transform.hpp"
#include "readMesh3d.hpp"
#include "vtkio.hpp"
#include "vulkanexamplebase.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kWorkgroupSize = 64;
constexpr float kEpsilon = 1.0e-8f;
// Keep the same fixed 60 Hz outer simulation step used by riddfmb3d.
// A rendered frame advances exactly this much simulation time; the step may
// be split into multiple XPBD substeps for stability.
constexpr float kSimulationFrameDeltaT = comparison_time_stepping::kFrameDeltaT;

uint32_t dispatchCount(uint32_t count) {
  return std::max(1u, (count + kWorkgroupSize - 1) / kWorkgroupSize);
}

template <int width, typename ConstraintAt>
void buildCorrectionAdjacency(uint32_t nParticles, uint32_t nConstraints,
                              ConstraintAt constraintAt,
                              std::vector<int32_t> &offsets,
                              std::vector<int32_t> &correctionIndices) {
  offsets.assign(nParticles + 1, 0);
  for (uint32_t constraint = 0; constraint < nConstraints; ++constraint) {
    for (uint32_t local = 0; local < width; ++local) {
      const int32_t particle = constraintAt(constraint, local);
      if (particle < 0 || static_cast<uint32_t>(particle) >= nParticles) {
        throw std::runtime_error(
            "Jacobi XPBD topology contains an invalid particle index");
      }
      ++offsets[static_cast<size_t>(particle) + 1];
    }
  }
  for (uint32_t particle = 0; particle < nParticles; ++particle) {
    offsets[particle + 1] += offsets[particle];
  }

  correctionIndices.resize(static_cast<size_t>(nConstraints) * width);
  auto cursor = offsets;
  for (uint32_t constraint = 0; constraint < nConstraints; ++constraint) {
    for (uint32_t local = 0; local < width; ++local) {
      const int32_t particle = constraintAt(constraint, local);
      correctionIndices[cursor[particle]++] =
          static_cast<int32_t>(constraint * width + local);
    }
  }
}

} // namespace

// A standalone PC example for the Distance + Volume Jacobi XPBD method used
// by vksim's DfmbXpbdJacobiStepper.  Unlike the older xpbddfmb3d sample, no
// constraint writes positions in-place: evaluate stages produce per-constraint
// corrections and apply stages gather those corrections per vertex.
class VulkanExample : public VulkanExampleBase {
public:
  struct Configuration {
    std::string modelPath{"models/beam3d.vtk"};
    std::string scene{"cantilever"};
    uint32_t iterations{1};
    uint32_t substeps{20};
    float density{1000.0f};
    float damping{1.5f};
    float youngsModulus{1000000.0f};
    float poissonRatio{0.40f};
    float gravityMagnitude{9.8f};
  } config;

  struct Particle {
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 uv;
    glm::vec4 normal;
  };

  // Explicit vec4 columns keep the CPU structure identical to std430 GLSL
  // mat3x4 storage, including its 16-byte column stride.
  struct alignas(16) ElementInfo {
    glm::ivec4 pid;
    glm::mat3x4 restShapeInv;
    glm::vec4 restData; // x = rest volume
  };

  struct alignas(16) EdgeInfo {
    glm::ivec4 pid;     // x/y = endpoints
    glm::vec4 restData; // x = rest length
  };

#if defined(XPBD_RID_SWITCH_DEMO)
  // This layout matches riddfmb3d/rid_*.comp exactly.
  struct alignas(16) RidElementInfo {
    int32_t elemId;
    float restVol;
    alignas(16) glm::ivec4 pid;
    alignas(16) glm::mat3x4 restShape;
    alignas(16) glm::mat3x4 restShapeInv;
  };

  struct alignas(16) RidSimulationParams {
    float deltaT{0.0f};
    float density{0.0f};
    float damping{0.0f};
    alignas(16) glm::vec4 gravity{0.0f};
    alignas(16) glm::vec4 lame{0.0f};
    glm::ivec2 particleCount{0};
    alignas(16) glm::vec4 ground{0.0f};
  };
#endif

  struct alignas(16) SimulationParams {
    float deltaT{0.0f};
    float relaxation{1.0f};
    float lameLambda{0.0f};
    float lameMu{0.0f};
    glm::vec4 gravity{0.0f};
    float damping{0.0f};
    uint32_t particleCount{0};
    uint32_t edgeCount{0};
    uint32_t elementCount{0};
  };

  // Matches the file-backed vksim beam-3pi protocol.  Each endpoint receives
  // half of the relative 3*pi twist, with opposing signs around the X axis.
  struct alignas(16) TwistBoundaryParams {
    glm::vec4 leftPivotAngle{0.0f};
    glm::vec4 rightPivotAngle{0.0f};
    // x is set when the prescribed 3*pi rotation has finished.  The twist
    // shader then turns every particle into a fixed point in the same substep.
    glm::vec4 control{0.0f};
  };

  static_assert(sizeof(ElementInfo) == 80,
                "ElementInfo must match std430 ElementInfo in the shaders");
  static_assert(sizeof(EdgeInfo) == 32,
                "EdgeInfo must match std430 EdgeInfo in the shaders");
  static_assert(sizeof(SimulationParams) == 48,
                "SimulationParams must match the std140 parameter block");
  static_assert(sizeof(TwistBoundaryParams) == 48,
                "TwistBoundaryParams must match the twist shader");
#if defined(XPBD_RID_SWITCH_DEMO)
  static_assert(sizeof(RidElementInfo) == 128,
                "RidElementInfo must match riddfmb3d's std430 layout");
  static_assert(offsetof(RidElementInfo, pid) == 16 &&
                    offsetof(RidElementInfo, restShape) == 32 &&
                    offsetof(RidElementInfo, restShapeInv) == 80,
                "RidElementInfo member offsets must match RID shaders");
  static_assert(sizeof(RidSimulationParams) == 80,
                "RidSimulationParams must match riddfmb3d's std140 layout");
  static_assert(offsetof(RidSimulationParams, gravity) == 16 &&
                    offsetof(RidSimulationParams, lame) == 32 &&
                    offsetof(RidSimulationParams, particleCount) == 48 &&
                    offsetof(RidSimulationParams, ground) == 64,
                "RidSimulationParams member offsets must match RID shaders");
#endif

  struct Storage {
    vks::Buffer particles;
    vks::Buffer indices;
  } storage;

  struct Mesh {
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi tets;
    Eigen::MatrixXi tris;
  } mesh;

  struct Graphics {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    struct UniformData {
      glm::mat4 projection;
      glm::mat4 modelview;
      glm::vec4 lightPos{-2.0f, 4.0f, -2.0f, 1.0f};
    } uniformData;
    std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;
  } graphics;

  struct Compute {
    VkCommandPool commandPool{VK_NULL_HANDLE};
    std::array<VkCommandBuffer, maxConcurrentFrames> commandBuffers{};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline begin{VK_NULL_HANDLE};
      VkPipeline distanceEvaluate{VK_NULL_HANDLE};
      VkPipeline distanceApply{VK_NULL_HANDLE};
      VkPipeline volumeEvaluate{VK_NULL_HANDLE};
      VkPipeline volumeApply{VK_NULL_HANDLE};
      VkPipeline end{VK_NULL_HANDLE};
    } pipelines;
    vks::Buffer parameters;
    vks::Buffer massesInv;
    vks::Buffer fixedPoints;
    vks::Buffer elements;
    vks::Buffer edges;
    vks::Buffer edgeCorrections;
    vks::Buffer volumeCorrections;
    vks::Buffer edgeOffsets;
    vks::Buffer edgeCorrectionIndices;
    vks::Buffer volumeOffsets;
    vks::Buffer volumeCorrectionIndices;
  } compute;

  // This is a solver-agnostic kinematic boundary pass, not an XPBD stage.
  struct TwistBoundaryCompute {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    vks::Buffer restPositions;
    vks::Buffer endpointKinds;
  } twistBoundary;

#if defined(XPBD_RID_SWITCH_DEMO)
  struct RidCompute {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline begin{VK_NULL_HANDLE};
    VkPipeline solve{VK_NULL_HANDLE};
    VkPipeline end{VK_NULL_HANDLE};
    vks::Buffer parameters;
    vks::Buffer lambdas;
    vks::Buffer elements;
    vks::Buffer parallelSlots;
    vks::Buffer masses;
  } ridCompute;
#endif

  VulkanExample() : VulkanExampleBase() {
#if defined(XPBD_RID_SWITCH_DEMO)
    title = "Jacobi XPBD (power measurement)";
#else
    title = "Jacobi XPBD: Distance + Volume";
#endif
    camera.type = Camera::CameraType::lookat;
    camera.setPerspective(60.0f, static_cast<float>(width) / height, 0.1f,
                          512.0f);
    camera.setRotation(glm::vec3(-25.0f, -40.0f, 0.0f));
    camera.setTranslation(glm::vec3(0.0f, 0.0f, -8.0f));
  }

  ~VulkanExample() override {
    if (device == VK_NULL_HANDLE) {
      return;
    }
    vkDeviceWaitIdle(device);

    storage.particles.destroy();
    storage.indices.destroy();
    for (auto &buffer : graphics.uniformBuffers) {
      buffer.destroy();
    }
    vkDestroyPipeline(device, graphics.pipeline, nullptr);
    vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);

    compute.parameters.destroy();
    compute.massesInv.destroy();
    compute.fixedPoints.destroy();
    compute.elements.destroy();
    compute.edges.destroy();
    compute.edgeCorrections.destroy();
    compute.volumeCorrections.destroy();
    compute.edgeOffsets.destroy();
    compute.edgeCorrectionIndices.destroy();
    compute.volumeOffsets.destroy();
    compute.volumeCorrectionIndices.destroy();
    vkDestroyPipeline(device, compute.pipelines.begin, nullptr);
    vkDestroyPipeline(device, compute.pipelines.distanceEvaluate, nullptr);
    vkDestroyPipeline(device, compute.pipelines.distanceApply, nullptr);
    vkDestroyPipeline(device, compute.pipelines.volumeEvaluate, nullptr);
    vkDestroyPipeline(device, compute.pipelines.volumeApply, nullptr);
    vkDestroyPipeline(device, compute.pipelines.end, nullptr);
    vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
    vkDestroyCommandPool(device, compute.commandPool, nullptr);

    twistBoundary.restPositions.destroy();
    twistBoundary.endpointKinds.destroy();
    vkDestroyPipeline(device, twistBoundary.pipeline, nullptr);
    vkDestroyPipelineLayout(device, twistBoundary.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, twistBoundary.descriptorSetLayout,
                                 nullptr);
#if defined(XPBD_RID_SWITCH_DEMO)
    ridCompute.parameters.destroy();
    ridCompute.lambdas.destroy();
    ridCompute.elements.destroy();
    ridCompute.parallelSlots.destroy();
    ridCompute.masses.destroy();
    vkDestroyPipeline(device, ridCompute.begin, nullptr);
    vkDestroyPipeline(device, ridCompute.solve, nullptr);
    vkDestroyPipeline(device, ridCompute.end, nullptr);
    vkDestroyPipelineLayout(device, ridCompute.pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, ridCompute.descriptorSetLayout,
                                 nullptr);
#endif
  }

  void prepare() override {
    VulkanExampleBase::prepare();
    comparison_time_stepping::Configuration jsonConfig{
        config.substeps, config.youngsModulus, config.poissonRatio,
        config.gravityMagnitude, config.scene};
    comparison_time_stepping::loadConfiguration(
        getAssetPath() + comparison_time_stepping::kConfigAssetPath,
        jsonConfig);
    config.substeps = jsonConfig.substepsPerFrame;
    config.youngsModulus = jsonConfig.youngsModulus;
    config.poissonRatio = jsonConfig.poissonRatio;
    config.gravityMagnitude = jsonConfig.gravityMagnitude;
    config.scene = jsonConfig.scene;
#if defined(XPBD_RID_SWITCH_DEMO)
    // Lock the measurement executable to XPBD. The RID backend is never
    // initialized or dispatched from this application.
    selectedSolver = 0;
#endif
    config.modelPath = isBunnySquashScene() ? "models/bunny_3828_asc.vtk"
                                            : "models/beam3d.vtk";
    // Let the fully flattened bunny be inspected before its first elastic
    // solve.  Other scenes retain their normal automatic start behavior.
    paused = isBunnySquashScene();
    loadMeshAndBuildSharedTopology();
    prepareStorageBuffers();
    prepareDescriptorPool();
    prepareGraphics();
    prepareTwistBoundaryCompute();
    prepareComputeCommandBuffers();
    prepareSelectedSolver();
    updateSimulationParams();
    prepared = true;
  }

  void render() override {
    if (!prepared) {
      return;
    }

    // prepareFrame updates the ImGui controls before the next command buffers
    // are recorded, so parameter edits affect the same rendered frame.
    VulkanExampleBase::prepareFrame();

    if (resetRequested) {
      resetParticleState();
      resetRequested = false;
    }

    prepareSelectedSolver();
    updateSimulationParams();
    if (!paused) {
      buildComputeCommandBuffer(simulationSubsteps());
      VkSubmitInfo computeSubmit = vks::initializers::submitInfo();
      computeSubmit.commandBufferCount = 1;
      computeSubmit.pCommandBuffers = &compute.commandBuffers[currentBuffer];
      // Compute and graphics use the same queue. Queue order, plus the final
      // compute-to-vertex barrier, makes the simulation state visible to draw.
      VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmit, VK_NULL_HANDLE));
      simulationTime += kSimulationFrameDeltaT;
    }

    updateGraphicsUniform();
    buildGraphicsCommandBuffer();
    VulkanExampleBase::submitFrame();
  }

  // Mesh export is intentionally not part of this real-time comparison sample.
  void writeMesh() override {}

  void OnUpdateUIOverlay(vks::UIOverlay *) override {
#if defined(XPBD_RID_SWITCH_DEMO)
    ImGui::TextUnformatted("Solver: Jacobi XPBD (power measurement)");
#else
    ImGui::TextUnformatted("Solver: Jacobi XPBD (distance + volume)");
#endif
#if defined(XPBD_RID_SWITCH_DEMO)
    if (selectedSolver == 1) {
      ImGui::Text("RID: %u tetrahedra, %u color sets",
                  static_cast<uint32_t>(ridElements.size()),
                  ridParallelSlots.empty()
                      ? 0u
                      : static_cast<uint32_t>(ridParallelSlots.size() - 1));
    } else {
      ImGui::Text("Jacobi: %u edges, %u tetrahedra",
                  static_cast<uint32_t>(edges.size()),
                  static_cast<uint32_t>(elements.size()));
    }
#else
    ImGui::Text("%u edges, %u tetrahedra", static_cast<uint32_t>(edges.size()),
                static_cast<uint32_t>(elements.size()));
#endif
    if (isTwistScene()) {
      ImGui::TextUnformatted(
          "Scene: beam twist 3pi (opposing X-end rotations)");
    } else if (isBunnySquashScene()) {
      ImGui::TextUnformatted(
          "Scene: bunny squash (initial current positions have z = 0)");
    } else {
      ImGui::TextUnformatted("Scene: cantilever beam (left X slab fixed)");
    }
    ImGui::Text("Fixed time: 1/60 s per frame, %u substeps (JSON)",
                simulationSubsteps());

    bool changed = false;
    float youngsExponent = std::log10(
        std::clamp(config.youngsModulus, 1.0e5f, 1.0e8f));
    if (ImGui::SliderFloat("Young's modulus log10(E)", &youngsExponent,
                           5.0f, 8.0f, "%.2f")) {
      config.youngsModulus = std::pow(10.0f, youngsExponent);
      changed = true;
    }
    ImGui::Text("E = %.3e", config.youngsModulus);
    changed |= ImGui::SliderFloat("Poisson ratio (Pr)", &config.poissonRatio,
                                  0.30f, 0.49f, "%.3f");
    changed |= ImGui::SliderFloat(
        "Gravity (downward)", &config.gravityMagnitude, 0.0f, 50.0f, "%.2f");
    if (changed) {
      resetRequested = true;
    }
    if (ImGui::Button(paused ? "Resume simulation" : "Pause simulation")) {
      paused = !paused;
    }
    if (ImGui::Button("Reset simulation")) {
      resetRequested = true;
    }
  }

private:
  uint32_t indexCount{0};
  bool resetRequested{false};
  std::vector<Particle> initialParticles;
  std::vector<ElementInfo> elements;
  std::vector<EdgeInfo> edges;
  std::vector<float> massesInv;
  std::vector<int32_t> fixedPoints;
  std::vector<int32_t> twistEndpointKinds;
  std::vector<int32_t> edgeOffsets;
  std::vector<int32_t> edgeCorrectionIndices;
  std::vector<int32_t> volumeOffsets;
  std::vector<int32_t> volumeCorrectionIndices;
  glm::vec3 twistLeftPivot{0.0f};
  glm::vec3 twistRightPivot{0.0f};
  float simulationTime{0.0f};
#if defined(XPBD_RID_SWITCH_DEMO)
  int selectedSolver{0}; // 0 = Jacobi, 1 = RID
  bool ridPrepared{false};
  std::vector<RidElementInfo> ridElements;
  std::vector<float> ridMasses;
  std::vector<float> ridLambdas;
  std::vector<int32_t> ridParallelSlots;
#endif
  bool jacobiPrepared{false};

  uint32_t simulationSubsteps() const { return std::max(1u, config.substeps); }

  bool isTwistScene() const { return config.scene == "beam_twist_3pi"; }

  bool isBunnySquashScene() const { return config.scene == "bunny_squash"; }

  float simulationSubstepDeltaT() const {
    return kSimulationFrameDeltaT / static_cast<float>(simulationSubsteps());
  }

  static glm::vec4 lameFromYoungsAndPoisson(float youngsModulus,
                                            float poissonRatio) {
    const float safeYoungs = std::max(youngsModulus, 1.0f);
    const float safePoisson = glm::clamp(poissonRatio, -0.99f, 0.499f);
    const float lambda = safeYoungs * safePoisson /
                         ((1.0f + safePoisson) * (1.0f - 2.0f * safePoisson));
    const float mu = safeYoungs / (2.0f * (1.0f + safePoisson));
    return glm::vec4(lambda, mu, 0.0f, 0.0f);
  }

  void loadMeshAndBuildSharedTopology() {
    io::readMesh3d(getAssetPath() + config.modelPath, mesh.vertices, mesh.tets);
    if (mesh.vertices.rows() == 0 || mesh.tets.rows() == 0) {
      throw std::runtime_error(
          "Simulation requires a non-empty tetrahedral mesh");
    }
    io::internal::extractBoundary(mesh.vertices, mesh.tets, mesh.tris);

    buildFixedPoints();
  }

  void buildJacobiTopology() {
    elements.clear();
    elements.reserve(mesh.tets.rows());
    for (int row = 0; row < mesh.tets.rows(); ++row) {
      const glm::ivec4 pid(mesh.tets(row, 0), mesh.tets(row, 1),
                           mesh.tets(row, 2), mesh.tets(row, 3));
      const glm::vec3 p0 = positionAt(pid.x);
      const glm::vec3 p1 = positionAt(pid.y);
      const glm::vec3 p2 = positionAt(pid.z);
      const glm::vec3 p3 = positionAt(pid.w);
      const glm::mat3 restShape(p1 - p0, p2 - p0, p3 - p0);
      const float determinant = glm::determinant(restShape);
      if (!std::isfinite(determinant) || std::abs(determinant) <= kEpsilon) {
        throw std::runtime_error(
            "Jacobi XPBD mesh contains a degenerate tetrahedron");
      }
      const glm::mat3 inverseRestShape = glm::inverse(restShape);
      ElementInfo element{};
      element.pid = pid;
      element.restShapeInv = glm::mat3x4(glm::vec4(inverseRestShape[0], 0.0f),
                                         glm::vec4(inverseRestShape[1], 0.0f),
                                         glm::vec4(inverseRestShape[2], 0.0f));
      element.restData =
          glm::vec4(std::abs(determinant) / 6.0f, 0.0f, 0.0f, 0.0f);
      elements.push_back(element);
    }

    buildEdgeTopology();
    buildJacobiMasses();
    const uint32_t particleCount = static_cast<uint32_t>(mesh.vertices.rows());
    buildCorrectionAdjacency<2>(
        particleCount, static_cast<uint32_t>(edges.size()),
        [&](uint32_t edge, uint32_t local) { return edges[edge].pid[local]; },
        edgeOffsets, edgeCorrectionIndices);
    buildCorrectionAdjacency<4>(
        particleCount, static_cast<uint32_t>(elements.size()),
        [&](uint32_t element, uint32_t local) {
          return elements[element].pid[local];
        },
        volumeOffsets, volumeCorrectionIndices);
  }

  glm::vec3 positionAt(int32_t vertex) const {
    return glm::vec3(mesh.vertices(vertex, 0), mesh.vertices(vertex, 1),
                     mesh.vertices(vertex, 2));
  }

  void buildEdgeTopology() {
    std::map<std::pair<int32_t, int32_t>, float> restLengths;
    constexpr std::array<std::array<int, 2>, 6> localEdges = {
        std::array<int, 2>{0, 1}, std::array<int, 2>{0, 2},
        std::array<int, 2>{0, 3}, std::array<int, 2>{1, 2},
        std::array<int, 2>{1, 3}, std::array<int, 2>{2, 3}};
    for (const auto &element : elements) {
      for (const auto &local : localEdges) {
        const int32_t a = element.pid[local[0]];
        const int32_t b = element.pid[local[1]];
        const auto key = std::minmax(a, b);
        const float length =
            glm::length(positionAt(key.second) - positionAt(key.first));
        if (!std::isfinite(length) || length <= kEpsilon) {
          throw std::runtime_error(
              "Jacobi XPBD mesh contains a zero-length edge");
        }
        restLengths.emplace(key, length);
      }
    }
    edges.clear();
    edges.reserve(restLengths.size());
    for (const auto &[endpoints, length] : restLengths) {
      EdgeInfo edge{};
      edge.pid = glm::ivec4(endpoints.first, endpoints.second, 0, 0);
      edge.restData = glm::vec4(length, 0.0f, 0.0f, 0.0f);
      edges.push_back(edge);
    }
  }

  void buildJacobiMasses() {
    const uint32_t particleCount = static_cast<uint32_t>(mesh.vertices.rows());
    std::vector<float> masses(particleCount, 0.0f);
    for (const auto &element : elements) {
      const float elementMass = config.density * element.restData.x;
      for (uint32_t local = 0; local < 4; ++local) {
        masses[element.pid[local]] += elementMass * 0.25f;
      }
    }
    massesInv.resize(particleCount);
    for (uint32_t particle = 0; particle < particleCount; ++particle) {
      massesInv[particle] =
          masses[particle] > kEpsilon ? 1.0f / masses[particle] : 0.0f;
    }
  }

  void buildFixedPoints() {
    const uint32_t particleCount = static_cast<uint32_t>(mesh.vertices.rows());
    float xMin = std::numeric_limits<float>::max();
    float xMax = std::numeric_limits<float>::lowest();
    for (uint32_t particle = 0; particle < particleCount; ++particle) {
      const float x = static_cast<float>(mesh.vertices(particle, 0));
      xMin = std::min(xMin, x);
      xMax = std::max(xMax, x);
    }
    fixedPoints.assign(particleCount, 0);
    twistEndpointKinds.assign(particleCount, 0);
    if (isBunnySquashScene()) {
      // The squash test is released from a fully flat current configuration.
      // It has no kinematic supports, so the elastic constraints alone drive
      // the bunny back toward its 3D rest shape.
      return;
    }
    if (!isTwistScene()) {
      const float fixedEnd = xMin + std::max((xMax - xMin) * 0.08f, 1.0e-4f);
      for (uint32_t particle = 0; particle < particleCount; ++particle) {
        fixedPoints[particle] = mesh.vertices(particle, 0) <= fixedEnd ? 1 : 0;
      }
      return;
    }

    // Exact endpoint slabs from vksim's frozen beam-3pi case.  beam3d.vtk
    // spans x=[0,6], so each selector contains one 6x6 endpoint section.
    constexpr float kEndpointThickness = 0.01f;
    glm::vec3 leftSum(0.0f);
    glm::vec3 rightSum(0.0f);
    uint32_t leftCount = 0;
    uint32_t rightCount = 0;
    for (uint32_t particle = 0; particle < particleCount; ++particle) {
      const glm::vec3 position = positionAt(static_cast<int32_t>(particle));
      if (position.x <= xMin + kEndpointThickness + kEpsilon) {
        twistEndpointKinds[particle] = 1;
        fixedPoints[particle] = 1;
        leftSum += position;
        ++leftCount;
      } else if (position.x >= xMax - kEndpointThickness - kEpsilon) {
        twistEndpointKinds[particle] = 2;
        fixedPoints[particle] = 1;
        rightSum += position;
        ++rightCount;
      }
    }
    if (leftCount == 0 || rightCount == 0) {
      throw std::runtime_error(
          "beam_twist_3pi requires particles at both beam endpoints");
    }
    twistLeftPivot = leftSum / static_cast<float>(leftCount);
    twistRightPivot = rightSum / static_cast<float>(rightCount);
  }

#if defined(XPBD_RID_SWITCH_DEMO)
  void buildRidTopology() {
    ridElements.clear();
    ridElements.reserve(static_cast<size_t>(mesh.tets.rows()));
    for (int row = 0; row < mesh.tets.rows(); ++row) {
      RidElementInfo element{};
      element.elemId = row;
      element.pid = glm::ivec4(mesh.tets(row, 0), mesh.tets(row, 1),
                               mesh.tets(row, 2), mesh.tets(row, 3));
      const glm::vec3 p0 = positionAt(element.pid.x);
      const glm::vec3 p1 = positionAt(element.pid.y);
      const glm::vec3 p2 = positionAt(element.pid.z);
      const glm::vec3 p3 = positionAt(element.pid.w);
      const glm::mat3 restShape(p1 - p0, p2 - p0, p3 - p0);
      element.restVol = std::abs(glm::determinant(restShape)) / 6.0f;
      const glm::mat3 restShapeInv = glm::inverse(restShape);
      element.restShape = glm::mat3x4(glm::vec4(restShape[0], 0.0f),
                                      glm::vec4(restShape[1], 0.0f),
                                      glm::vec4(restShape[2], 0.0f));
      element.restShapeInv = glm::mat3x4(glm::vec4(restShapeInv[0], 0.0f),
                                         glm::vec4(restShapeInv[1], 0.0f),
                                         glm::vec4(restShapeInv[2], 0.0f));
      ridElements.push_back(element);
    }

    std::vector<int32_t> remaining(ridElements.size());
    for (uint32_t i = 0; i < remaining.size(); ++i) {
      remaining[i] = static_cast<int32_t>(i);
    }
    std::vector<RidElementInfo> ordered;
    ordered.reserve(ridElements.size());
    ridParallelSlots.clear();
    while (!remaining.empty()) {
      std::vector<bool> occupied(static_cast<size_t>(mesh.vertices.rows()),
                                 false);
      ridParallelSlots.push_back(static_cast<int32_t>(ordered.size()));
      for (auto it = remaining.begin(); it != remaining.end();) {
        const RidElementInfo &element = ridElements[*it];
        bool conflict = false;
        for (uint32_t local = 0; local < 4; ++local) {
          conflict |= occupied[element.pid[local]];
        }
        if (conflict) {
          ++it;
          continue;
        }
        RidElementInfo copy = element;
        copy.elemId = static_cast<int32_t>(ordered.size());
        ordered.push_back(copy);
        for (uint32_t local = 0; local < 4; ++local) {
          occupied[element.pid[local]] = true;
        }
        it = remaining.erase(it);
      }
    }
    ridParallelSlots.push_back(static_cast<int32_t>(ordered.size()));
    ridElements = std::move(ordered);

    ridMasses.assign(static_cast<size_t>(mesh.vertices.rows()), 0.0f);
    for (const RidElementInfo &element : ridElements) {
      const float elementMass = config.density * element.restVol;
      for (uint32_t local = 0; local < 4; ++local) {
        ridMasses[element.pid[local]] += elementMass * 0.25f;
      }
    }
    ridLambdas.assign(ridElements.size(), 0.0f);
  }
#endif

  void uploadDeviceLocalBuffer(vks::Buffer &destination,
                               VkBufferUsageFlags usage, const void *data,
                               VkDeviceSize size) {
    vks::Buffer staging;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &staging, size, const_cast<void *>(data));
    vulkanDevice->createBuffer(usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &destination, size);
    VkCommandBuffer copyCommand = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(copyCommand, staging.buffer, destination.buffer, 1, &copy);
    VkBufferMemoryBarrier copyBarrier{};
    copyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copyBarrier.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.buffer = destination.buffer;
    copyBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(copyCommand, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1,
                         &copyBarrier, 0, nullptr);
    vulkanDevice->flushCommandBuffer(copyCommand, queue, true);
    staging.destroy();
  }

  void createDeviceLocalStorage(vks::Buffer &destination, VkDeviceSize size) {
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &destination, size);
  }

  void prepareStorageBuffers() {
    const uint32_t particleCount = static_cast<uint32_t>(mesh.vertices.rows());
    initialParticles.resize(particleCount);
    for (uint32_t particle = 0; particle < particleCount; ++particle) {
      Particle value{};
      glm::vec3 initialPosition = positionAt(static_cast<int32_t>(particle));
      if (isBunnySquashScene()) {
        // Keep all rest data from the 3D VTK mesh, but start the dynamic
        // configuration completely flattened onto the z = 0 plane.
        initialPosition.z = 0.0f;
      }
      value.pos = glm::vec4(initialPosition, 1.0f);
      value.vel = glm::vec4(0.0f);
      value.uv = glm::vec4(0.0f);
      value.normal = glm::vec4(0.0f);
      initialParticles[particle] = value;
    }
    for (int tri = 0; tri < mesh.tris.rows(); ++tri) {
      const uint32_t a = static_cast<uint32_t>(mesh.tris(tri, 0));
      const uint32_t b = static_cast<uint32_t>(mesh.tris(tri, 1));
      const uint32_t c = static_cast<uint32_t>(mesh.tris(tri, 2));
      // Normals describe the 3D rest surface.  In bunny_squash the current
      // positions are coplanar initially, so deriving normals from them would
      // lose the surface orientation before the elastic recovery starts.
      const glm::vec3 normal =
          glm::cross(positionAt(static_cast<int32_t>(b)) -
                         positionAt(static_cast<int32_t>(a)),
                     positionAt(static_cast<int32_t>(c)) -
                         positionAt(static_cast<int32_t>(a)));
      initialParticles[a].normal += glm::vec4(normal, 0.0f);
      initialParticles[b].normal += glm::vec4(normal, 0.0f);
      initialParticles[c].normal += glm::vec4(normal, 0.0f);
    }
    for (auto &particle : initialParticles) {
      const float normalLength = glm::length(glm::vec3(particle.normal));
      if (normalLength > kEpsilon) {
        particle.normal /= normalLength;
      }
    }

    uploadDeviceLocalBuffer(
        storage.particles,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        initialParticles.data(), initialParticles.size() * sizeof(Particle));

    std::vector<uint32_t> triangleIndices(
        static_cast<size_t>(mesh.tris.rows()) * 3);
    for (int tri = 0; tri < mesh.tris.rows(); ++tri) {
      for (int local = 0; local < 3; ++local) {
        triangleIndices[static_cast<size_t>(tri) * 3 + local] =
            static_cast<uint32_t>(mesh.tris(tri, local));
      }
    }
    indexCount = static_cast<uint32_t>(triangleIndices.size());
    uploadDeviceLocalBuffer(storage.indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            triangleIndices.data(),
                            triangleIndices.size() * sizeof(uint32_t));

    uploadDeviceLocalBuffer(
        compute.fixedPoints, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        fixedPoints.data(), fixedPoints.size() * sizeof(int32_t));
    std::vector<glm::vec4> restPositions(particleCount);
    for (uint32_t particle = 0; particle < particleCount; ++particle) {
      restPositions[particle] = initialParticles[particle].pos;
    }
    uploadDeviceLocalBuffer(
        twistBoundary.restPositions, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        restPositions.data(), restPositions.size() * sizeof(glm::vec4));
    uploadDeviceLocalBuffer(
        twistBoundary.endpointKinds, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        twistEndpointKinds.data(), twistEndpointKinds.size() * sizeof(int32_t));
  }

  void prepareJacobiStorageBuffers() {
    uploadDeviceLocalBuffer(compute.massesInv,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            massesInv.data(), massesInv.size() * sizeof(float));
    uploadDeviceLocalBuffer(compute.elements,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, elements.data(),
                            elements.size() * sizeof(ElementInfo));
    uploadDeviceLocalBuffer(compute.edges, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            edges.data(), edges.size() * sizeof(EdgeInfo));
    uploadDeviceLocalBuffer(
        compute.edgeOffsets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        edgeOffsets.data(), edgeOffsets.size() * sizeof(int32_t));
    uploadDeviceLocalBuffer(compute.edgeCorrectionIndices,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            edgeCorrectionIndices.data(),
                            edgeCorrectionIndices.size() * sizeof(int32_t));
    uploadDeviceLocalBuffer(
        compute.volumeOffsets, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        volumeOffsets.data(), volumeOffsets.size() * sizeof(int32_t));
    uploadDeviceLocalBuffer(compute.volumeCorrectionIndices,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            volumeCorrectionIndices.data(),
                            volumeCorrectionIndices.size() * sizeof(int32_t));
    createDeviceLocalStorage(compute.edgeCorrections,
                             edges.size() * 2 * sizeof(glm::vec4));
    createDeviceLocalStorage(compute.volumeCorrections,
                             elements.size() * 4 * sizeof(glm::vec4));
  }

#if defined(XPBD_RID_SWITCH_DEMO)
  void prepareRidStorageBuffers() {
    uploadDeviceLocalBuffer(
        ridCompute.lambdas, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        ridLambdas.data(), ridLambdas.size() * sizeof(float));
    uploadDeviceLocalBuffer(
        ridCompute.elements, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        ridElements.data(), ridElements.size() * sizeof(RidElementInfo));
    uploadDeviceLocalBuffer(
        ridCompute.parallelSlots, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        ridParallelSlots.data(), ridParallelSlots.size() * sizeof(int32_t));
    uploadDeviceLocalBuffer(ridCompute.masses,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            ridMasses.data(), ridMasses.size() * sizeof(float));
  }
#endif

  void prepareDescriptorPool() {
    const std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames +
#if defined(XPBD_RID_SWITCH_DEMO)
                                                  2),
#else
                                                  1),
#endif
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              15
#if defined(XPBD_RID_SWITCH_DEMO)
                                                  + 6
#endif
                                              )};
    const VkDescriptorPoolCreateInfo poolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    maxConcurrentFrames +
#if defined(XPBD_RID_SWITCH_DEMO)
                                                        3);
#else
                                                        2);
#endif
    VK_CHECK_RESULT(
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
  }

  void prepareGraphics() {
    for (auto &buffer : graphics.uniformBuffers) {
      vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &buffer, sizeof(Graphics::UniformData));
      VK_CHECK_RESULT(buffer.map());
    }
    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0)};
    VkDescriptorSetLayoutCreateInfo layoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(bindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                                &graphics.descriptorSetLayout));
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &graphics.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                           &graphics.pipelineLayout));
    for (uint32_t frame = 0; frame < maxConcurrentFrames; ++frame) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &graphics.descriptorSetLayout, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(
          device, &allocInfo, &graphics.descriptorSets[frame]));
      const VkWriteDescriptorSet write = vks::initializers::writeDescriptorSet(
          graphics.descriptorSets[frame], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
          &graphics.uniformBuffers[frame].descriptor);
      vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        loadShader(getShadersPath() + "xpbdjacobi3d/beam3d.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "xpbdjacobi3d/beam3d.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT)};
    const std::vector<VkVertexInputBindingDescription> inputBindings = {
        vks::initializers::vertexInputBindingDescription(
            0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)};
    const std::vector<VkVertexInputAttributeDescription> inputAttributes = {
        vks::initializers::vertexInputAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, pos)),
        vks::initializers::vertexInputAttributeDescription(
            0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, uv)),
        vks::initializers::vertexInputAttributeDescription(
            0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, normal))};
    VkPipelineVertexInputStateCreateInfo vertexInput =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInput.vertexBindingDescriptionCount =
        static_cast<uint32_t>(inputBindings.size());
    vertexInput.pVertexBindingDescriptions = inputBindings.data();
    vertexInput.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(inputAttributes.size());
    vertexInput.pVertexAttributeDescriptions = inputAttributes.data();
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    const VkPipelineRasterizationStateCreateInfo rasterization =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    const VkPipelineColorBlendAttachmentState blendAttachment =
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    const VkPipelineColorBlendStateCreateInfo blend =
        vks::initializers::pipelineColorBlendStateCreateInfo(1,
                                                             &blendAttachment);
    const VkPipelineDepthStencilStateCreateInfo depthStencil =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    const VkPipelineViewportStateCreateInfo viewport =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    const VkPipelineMultisampleStateCreateInfo multisample =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT, 0);
    const std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStates);
    VkGraphicsPipelineCreateInfo pipelineInfo =
        vks::initializers::pipelineCreateInfo(graphics.pipelineLayout,
                                              renderPass);
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pDynamicState = &dynamic;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(
        device, pipelineCache, 1, &pipelineInfo, nullptr, &graphics.pipeline));
  }

  void prepareComputeCommandBuffers() {
    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex =
        vulkanDevice->queueFamilyIndices.graphics;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolInfo, nullptr,
                                        &compute.commandPool));
    const VkCommandBufferAllocateInfo commandBufferInfo =
        vks::initializers::commandBufferAllocateInfo(
            compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            static_cast<uint32_t>(compute.commandBuffers.size()));
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferInfo,
                                             compute.commandBuffers.data()));
  }

  void prepareCompute() {
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &compute.parameters, sizeof(SimulationParams));
    VK_CHECK_RESULT(compute.parameters.map());

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(12);
    for (uint32_t binding = 0; binding < 12; ++binding) {
      const VkDescriptorType type = binding == 1
                                        ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings.push_back(vks::initializers::descriptorSetLayoutBinding(
          type, VK_SHADER_STAGE_COMPUTE_BIT, binding));
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(bindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                                &compute.descriptorSetLayout));
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                           &compute.pipelineLayout));
    VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool, &compute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(
        vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));
    const std::array<VkDescriptorBufferInfo *, 12> descriptors = {
        &storage.particles.descriptor,
        &compute.parameters.descriptor,
        &compute.massesInv.descriptor,
        &compute.fixedPoints.descriptor,
        &compute.elements.descriptor,
        &compute.edges.descriptor,
        &compute.edgeCorrections.descriptor,
        &compute.volumeCorrections.descriptor,
        &compute.edgeOffsets.descriptor,
        &compute.edgeCorrectionIndices.descriptor,
        &compute.volumeOffsets.descriptor,
        &compute.volumeCorrectionIndices.descriptor};
    std::array<VkWriteDescriptorSet, 12> writes{};
    for (uint32_t binding = 0; binding < writes.size(); ++binding) {
      writes[binding] = vks::initializers::writeDescriptorSet(
          compute.descriptorSet,
          binding == 1 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          binding, descriptors[binding]);
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    VkComputePipelineCreateInfo pipelineInfo =
        vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
    const auto makePipeline = [&](const char *name, VkPipeline &pipeline) {
      pipelineInfo.stage = loadShader(getShadersPath() + "xpbdjacobi3d/" + name,
                                      VK_SHADER_STAGE_COMPUTE_BIT);
      VK_CHECK_RESULT(vkCreateComputePipelines(
          device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline));
    };
    makePipeline("jacobi_begin.comp.spv", compute.pipelines.begin);
    makePipeline("jacobi_distance_evaluate.comp.spv",
                 compute.pipelines.distanceEvaluate);
    makePipeline("jacobi_distance_apply.comp.spv",
                 compute.pipelines.distanceApply);
    makePipeline("jacobi_volume_evaluate.comp.spv",
                 compute.pipelines.volumeEvaluate);
    makePipeline("jacobi_volume_apply.comp.spv", compute.pipelines.volumeApply);
    makePipeline("jacobi_end.comp.spv", compute.pipelines.end);
  }

  void prepareJacobiSolver() {
    if (jacobiPrepared) {
      return;
    }
    buildJacobiTopology();
    prepareJacobiStorageBuffers();
    prepareCompute();
    jacobiPrepared = true;
  }

#if defined(XPBD_RID_SWITCH_DEMO)
  void prepareRidSolver() {
    if (ridPrepared) {
      return;
    }
    buildRidTopology();
    prepareRidStorageBuffers();
    prepareRidCompute();
    ridPrepared = true;
  }
#endif

  void prepareSelectedSolver() {
#if defined(XPBD_RID_SWITCH_DEMO)
    if (selectedSolver == 1) {
      prepareRidSolver();
      return;
    }
#endif
    prepareJacobiSolver();
  }

  void prepareTwistBoundaryCompute() {
    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3)};
    const VkDescriptorSetLayoutCreateInfo layoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(bindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &layoutInfo, nullptr, &twistBoundary.descriptorSetLayout));

    const VkPushConstantRange pushRange = vks::initializers::pushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(TwistBoundaryParams), 0);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &twistBoundary.descriptorSetLayout, 1);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                           &twistBoundary.pipelineLayout));

    const VkDescriptorSetAllocateInfo allocateInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool, &twistBoundary.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocateInfo,
                                             &twistBoundary.descriptorSet));
    const std::array<VkWriteDescriptorSet, 4> writes = {
        vks::initializers::writeDescriptorSet(twistBoundary.descriptorSet,
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              0, &storage.particles.descriptor),
        vks::initializers::writeDescriptorSet(
            twistBoundary.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            &twistBoundary.restPositions.descriptor),
        vks::initializers::writeDescriptorSet(
            twistBoundary.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2,
            &twistBoundary.endpointKinds.descriptor),
        vks::initializers::writeDescriptorSet(
            twistBoundary.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3,
            &compute.fixedPoints.descriptor)};
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    VkComputePipelineCreateInfo pipelineInfo =
        vks::initializers::computePipelineCreateInfo(
            twistBoundary.pipelineLayout, 0);
    pipelineInfo.stage =
        loadShader(getShadersPath() + "base/twist_boundary.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1,
                                             &pipelineInfo, nullptr,
                                             &twistBoundary.pipeline));
  }

  void updateSimulationParams() {
    const glm::vec4 lame =
        lameFromYoungsAndPoisson(config.youngsModulus, config.poissonRatio);
#if defined(XPBD_RID_SWITCH_DEMO)
    if (selectedSolver == 1) {
      RidSimulationParams ridParams{};
      ridParams.deltaT = simulationSubstepDeltaT();
      ridParams.density = config.density;
      ridParams.damping = config.damping;
      ridParams.gravity = glm::vec4(0.0f, -config.gravityMagnitude, 0.0f, 0.0f);
      ridParams.lame = lame;
      ridParams.particleCount =
          glm::ivec2(static_cast<int32_t>(initialParticles.size()), 1);
      // The common comparison scene deliberately has no ground collision.
      ridParams.ground = glm::vec4(0.0f);
      std::memcpy(ridCompute.parameters.mapped, &ridParams, sizeof(ridParams));
      return;
    }
#endif

    SimulationParams params{};
    params.deltaT = simulationSubstepDeltaT();
    params.relaxation = 1.0f;
    params.lameLambda = lame.x;
    params.lameMu = lame.y;
    params.gravity = glm::vec4(0.0f, -config.gravityMagnitude, 0.0f, 0.0f);
    params.damping = config.damping;
    params.particleCount = static_cast<uint32_t>(initialParticles.size());
    params.edgeCount = static_cast<uint32_t>(edges.size());
    params.elementCount = static_cast<uint32_t>(elements.size());
    std::memcpy(compute.parameters.mapped, &params, sizeof(params));
  }

#if defined(XPBD_RID_SWITCH_DEMO)
  void prepareRidCompute() {
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &ridCompute.parameters,
                               sizeof(RidSimulationParams));
    VK_CHECK_RESULT(ridCompute.parameters.map());

    const std::vector<VkDescriptorSetLayoutBinding> bindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8)};
    const VkDescriptorSetLayoutCreateInfo layoutInfo =
        vks::initializers::descriptorSetLayoutCreateInfo(bindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &layoutInfo, nullptr, &ridCompute.descriptorSetLayout));
    VkPushConstantRange pushRange = vks::initializers::pushConstantRange(
        VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t), 0);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &ridCompute.descriptorSetLayout, 1);
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                                           &ridCompute.pipelineLayout));
    const VkDescriptorSetAllocateInfo allocateInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool, &ridCompute.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocateInfo,
                                             &ridCompute.descriptorSet));
    const std::array<VkWriteDescriptorSet, 7> writes = {
        vks::initializers::writeDescriptorSet(ridCompute.descriptorSet,
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              0, &storage.particles.descriptor),
        vks::initializers::writeDescriptorSet(
            ridCompute.descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
            &ridCompute.parameters.descriptor),
        vks::initializers::writeDescriptorSet(
            ridCompute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3,
            &ridCompute.lambdas.descriptor),
        vks::initializers::writeDescriptorSet(
            ridCompute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
            &ridCompute.elements.descriptor),
        vks::initializers::writeDescriptorSet(
            ridCompute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5,
            &ridCompute.parallelSlots.descriptor),
        vks::initializers::writeDescriptorSet(ridCompute.descriptorSet,
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              6, &ridCompute.masses.descriptor),
        vks::initializers::writeDescriptorSet(
            ridCompute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8,
            &compute.fixedPoints.descriptor)};
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);

    VkComputePipelineCreateInfo pipelineInfo =
        vks::initializers::computePipelineCreateInfo(ridCompute.pipelineLayout,
                                                     0);
    const auto makePipeline = [&](const char *name, VkPipeline &pipeline) {
      pipelineInfo.stage = loadShader(getShadersPath() + "riddfmb3d/" + name,
                                      VK_SHADER_STAGE_COMPUTE_BIT);
      VK_CHECK_RESULT(vkCreateComputePipelines(
          device, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline));
    };
    makePipeline("rid_begin.comp.spv", ridCompute.begin);
    makePipeline("rid_solve.comp.spv", ridCompute.solve);
    makePipeline("rid_end.comp.spv", ridCompute.end);
  }
#endif

  void updateGraphicsUniform() {
    graphics.uniformData.projection = camera.matrices.perspective;
    const glm::mat4 flipY =
        glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, 1.0f));
    graphics.uniformData.modelview = camera.matrices.view * flipY;
    std::memcpy(graphics.uniformBuffers[currentBuffer].mapped,
                &graphics.uniformData, sizeof(graphics.uniformData));
  }

  void addComputeBarrier(VkCommandBuffer commandBuffer) const {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  }

  void buildJacobiComputeCommandBuffer(uint32_t substeps) {
    VkCommandBuffer commandBuffer = compute.commandBuffers[currentBuffer];
    VK_CHECK_RESULT(vkResetCommandBuffer(commandBuffer, 0));
    const VkCommandBufferBeginInfo beginInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferMemoryBarrier graphicsReadBarrier =
        vks::initializers::bufferMemoryBarrier();
    graphicsReadBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    graphicsReadBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    graphicsReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    graphicsReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    graphicsReadBarrier.buffer = storage.particles.buffer;
    graphicsReadBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &graphicsReadBarrier, 0, nullptr);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSet, 0, nullptr);
    const uint32_t particleGroups =
        dispatchCount(static_cast<uint32_t>(initialParticles.size()));
    const uint32_t edgeGroups =
        dispatchCount(static_cast<uint32_t>(edges.size()));
    const uint32_t elementGroups =
        dispatchCount(static_cast<uint32_t>(elements.size()));
    for (uint32_t substep = 0; substep < substeps; ++substep) {
      applyTwistBoundary(commandBuffer,
                         simulationTime + static_cast<float>(substep + 1) *
                                              simulationSubstepDeltaT());
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              compute.pipelineLayout, 0, 1,
                              &compute.descriptorSet, 0, nullptr);
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.begin);
      vkCmdDispatch(commandBuffer, particleGroups, 1, 1);
      addComputeBarrier(commandBuffer);

      for (uint32_t iteration = 0; iteration < config.iterations; ++iteration) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          compute.pipelines.distanceEvaluate);
        vkCmdDispatch(commandBuffer, edgeGroups, 1, 1);
        addComputeBarrier(commandBuffer);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          compute.pipelines.distanceApply);
        vkCmdDispatch(commandBuffer, particleGroups, 1, 1);
        addComputeBarrier(commandBuffer);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          compute.pipelines.volumeEvaluate);
        vkCmdDispatch(commandBuffer, elementGroups, 1, 1);
        addComputeBarrier(commandBuffer);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          compute.pipelines.volumeApply);
        vkCmdDispatch(commandBuffer, particleGroups, 1, 1);
        addComputeBarrier(commandBuffer);
      }

      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.end);
      vkCmdDispatch(commandBuffer, particleGroups, 1, 1);
      if (substep + 1 < substeps) {
        addComputeBarrier(commandBuffer);
      }
    }

    VkBufferMemoryBarrier drawBarrier =
        vks::initializers::bufferMemoryBarrier();
    drawBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    drawBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    drawBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    drawBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    drawBarrier.buffer = storage.particles.buffer;
    drawBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1,
                         &drawBarrier, 0, nullptr);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  void applyTwistBoundary(VkCommandBuffer commandBuffer, float time) const {
    if (!isTwistScene()) {
      return;
    }

    constexpr float kEndpointAngle = 4.71238898038469f; // 3*pi/2
    constexpr float kDriveTime = 2.356194490192345f;    // 3*pi/4 at 2 rad/s
    const float alpha = glm::clamp(time / kDriveTime, 0.0f, 1.0f);
    const TwistBoundaryParams params{
        glm::vec4(twistLeftPivot, alpha * kEndpointAngle),
        glm::vec4(twistRightPivot, -alpha * kEndpointAngle),
        glm::vec4(time >= kDriveTime ? 1.0f : 0.0f)};
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      twistBoundary.pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            twistBoundary.pipelineLayout, 0, 1,
                            &twistBoundary.descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, twistBoundary.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(TwistBoundaryParams), &params);
    vkCmdDispatch(commandBuffer,
                  dispatchCount(static_cast<uint32_t>(initialParticles.size())),
                  1, 1);
    addComputeBarrier(commandBuffer);
  }

#if defined(XPBD_RID_SWITCH_DEMO)
  void buildRidComputeCommandBuffer(uint32_t substeps) {
    VkCommandBuffer commandBuffer = compute.commandBuffers[currentBuffer];
    VK_CHECK_RESULT(vkResetCommandBuffer(commandBuffer, 0));
    const VkCommandBufferBeginInfo beginInfo =
        vks::initializers::commandBufferBeginInfo();
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferMemoryBarrier graphicsReadBarrier =
        vks::initializers::bufferMemoryBarrier();
    graphicsReadBarrier.srcAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    graphicsReadBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    graphicsReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    graphicsReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    graphicsReadBarrier.buffer = storage.particles.buffer;
    graphicsReadBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &graphicsReadBarrier, 0, nullptr);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ridCompute.pipelineLayout, 0, 1,
                            &ridCompute.descriptorSet, 0, nullptr);
    const uint32_t particleGroups =
        dispatchCount(static_cast<uint32_t>(initialParticles.size()));
    const uint32_t beginGroups =
        dispatchCount(std::max(static_cast<uint32_t>(initialParticles.size()),
                               static_cast<uint32_t>(ridElements.size())));
    const uint32_t colorCount =
        static_cast<uint32_t>(ridParallelSlots.size() - 1);
    for (uint32_t substep = 0; substep < substeps; ++substep) {
      applyTwistBoundary(commandBuffer,
                         simulationTime + static_cast<float>(substep + 1) *
                                              simulationSubstepDeltaT());
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              ridCompute.pipelineLayout, 0, 1,
                              &ridCompute.descriptorSet, 0, nullptr);
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        ridCompute.begin);
      vkCmdDispatch(commandBuffer, beginGroups, 1, 1);
      addComputeBarrier(commandBuffer);

      for (uint32_t iteration = 0; iteration < config.iterations; ++iteration) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          ridCompute.solve);
        for (uint32_t color = 0; color < colorCount; ++color) {
          vkCmdPushConstants(commandBuffer, ridCompute.pipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t),
                             &color);
          const uint32_t start = static_cast<uint32_t>(ridParallelSlots[color]);
          const uint32_t end =
              static_cast<uint32_t>(ridParallelSlots[color + 1]);
          vkCmdDispatch(commandBuffer, dispatchCount(end - start), 1, 1);
          addComputeBarrier(commandBuffer);
        }
      }

      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        ridCompute.end);
      vkCmdDispatch(commandBuffer, particleGroups, 1, 1);
      if (substep + 1 < substeps) {
        addComputeBarrier(commandBuffer);
      }
    }

    VkBufferMemoryBarrier drawBarrier =
        vks::initializers::bufferMemoryBarrier();
    drawBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    drawBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    drawBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    drawBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    drawBarrier.buffer = storage.particles.buffer;
    drawBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1,
                         &drawBarrier, 0, nullptr);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }
#endif

  void buildComputeCommandBuffer(uint32_t substeps) {
#if defined(XPBD_RID_SWITCH_DEMO)
    if (selectedSolver == 1) {
      buildRidComputeCommandBuffer(substeps);
      return;
    }
#endif
    buildJacobiComputeCommandBuffer(substeps);
  }

  void buildGraphicsCommandBuffer() {
    VkCommandBuffer commandBuffer = drawCmdBuffers[currentBuffer];
    const VkCommandBufferBeginInfo beginInfo =
        vks::initializers::commandBufferBeginInfo();
    VkClearValue clearValues[2]{};
    clearValues[0].color = defaultClearColor;
    clearValues[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo renderPassInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.renderArea.extent = {width, height};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;
    renderPassInfo.framebuffer = frameBuffers[currentImageIndex];
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = vks::initializers::viewport(
        static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics.pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout,
        0, 1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &storage.particles.buffer,
                           &offset);
    vkCmdBindIndexBuffer(commandBuffer, storage.indices.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
    drawUI(commandBuffer);
    vkCmdEndRenderPass(commandBuffer);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
  }

  void resetParticleState() {
    vkDeviceWaitIdle(device);
    simulationTime = 0.0f;
    // Reset returns bunny_squash to its inspectable flattened initial state.
    if (isBunnySquashScene()) {
      paused = true;
    }
    vks::Buffer staging;
    const VkDeviceSize size = initialParticles.size() * sizeof(Particle);
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &staging, size, initialParticles.data());
    VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    const VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(commandBuffer, staging.buffer, storage.particles.buffer, 1,
                    &copy);
    VkBufferMemoryBarrier copyBarrier{};
    copyBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copyBarrier.buffer = storage.particles.buffer;
    copyBarrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                         0, 0, nullptr, 1, &copyBarrier, 0, nullptr);
    vulkanDevice->flushCommandBuffer(commandBuffer, queue, true);
    staging.destroy();

    // A finished twist promotes every particle to a fixed point on the GPU.
    // Reset restores the scene's original endpoint-only fixed-point mask.
    vks::Buffer fixedPointsStaging;
    const VkDeviceSize fixedPointsSize =
        fixedPoints.size() * sizeof(fixedPoints.front());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &fixedPointsStaging, fixedPointsSize,
                               fixedPoints.data());
    commandBuffer = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    const VkBufferCopy fixedPointsCopy{0, 0, fixedPointsSize};
    vkCmdCopyBuffer(commandBuffer, fixedPointsStaging.buffer,
                    compute.fixedPoints.buffer, 1, &fixedPointsCopy);
    copyBarrier.buffer = compute.fixedPoints.buffer;
    copyBarrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
                         &copyBarrier, 0, nullptr);
    vulkanDevice->flushCommandBuffer(commandBuffer, queue, true);
    fixedPointsStaging.destroy();
  }
};

VULKAN_EXAMPLE_MAIN()
