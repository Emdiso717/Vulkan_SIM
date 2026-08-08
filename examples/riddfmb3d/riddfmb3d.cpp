
#include "../config.hpp"
#include "../vulkan_barriers.hpp"
#include "VulkanglTFModel.h"
#include "glm/fwd.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "vtkio.hpp"
#include "vulkanexamplebase.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <readMesh3d.hpp>

class VulkanExample : public VulkanExampleBase {
public:
  uint32_t indexCount{0};
  bool dedicatedComputeQueue{false};
  bool computeOnlyBuffersAcquired{false};
  bool writeVtk{false};
  bool readbackPending{false};
  uint32_t readbackFenceIndex{0};
  uint32_t readbackFrame{0};
  uint32_t Framecount{0};
  example_config::Riddfmb3dConfiguration config;

  struct Particle {
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 uv;
    glm::vec4 normal;
  };

  struct ElementInfo {
    int elemId;
    float restVol;
    alignas(16) glm::ivec4 pid;
    alignas(16) glm::mat3x4 restShape;
    alignas(16) glm::mat3x4 restShapeInv;
  };

  struct StorageBuffers {
    vks::Buffer particles;
  } storageBuffers;

  // Readback buffer for CPU export (GPU writes particles -> copy -> mapped)
  vks::Buffer outputReadback;
  VkDeviceSize storageBufferSize = 0;

  struct PushConstants {
    uint32_t parallelSetStartIndex;
  };

  struct Graphics {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline beam3d{VK_NULL_HANDLE};
    } pipelines;
    vks::Buffer indices;
    struct UniformData {
      glm::mat4 projection;
      glm::mat4 view;
      glm::vec4 lightPos{-2.0f, 4.0f, -2.0f, 1.0f};
    } uniformData;
    std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;
  } graphics;

  struct Compute {
    struct ComputeSemaphores {
      VkSemaphore ready{VK_NULL_HANDLE};
      VkSemaphore complete{VK_NULL_HANDLE};
    };
    std::array<ComputeSemaphores, maxConcurrentFrames> semaphores{};
    std::array<VkFence, maxConcurrentFrames> fences{};
    VkQueue queue{VK_NULL_HANDLE};
    VkCommandPool commandPool{VK_NULL_HANDLE};
    std::array<VkCommandBuffer, maxConcurrentFrames> commandBuffers{};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{
        VK_NULL_HANDLE};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline begin{VK_NULL_HANDLE};
      VkPipeline solve{VK_NULL_HANDLE};
      VkPipeline end{VK_NULL_HANDLE};
    } pipelines;
    struct UniformData {
      float deltaT{0.0f};
      float density{0.0f};
      alignas(16) glm::vec4 gravity{0.0f};
      glm::vec4 lame{0.0f};
      glm::ivec2 particleCount{0};
    } uniformData;
    std::vector<float> masses{};
    std::vector<ElementInfo> elementInfo;
    std::vector<float> lambdaHData;
    std::vector<float> lambdaDData;
    std::vector<int> elemParallelSlots;
    vks::Buffer uniformBuffer;
    vks::Buffer lambdaHBuffer;
    vks::Buffer lambdaDBuffer;
    vks::Buffer elementInfoBuffer;
    vks::Buffer elemParallelSlotsBuffer;
    vks::Buffer massesBuffer;
    std::vector<int> fixedpoint;
    vks::Buffer fixedpointBuffer;
  } compute;

  struct Mesh {
    Eigen::MatrixXd V;
    Eigen::MatrixXi tets;
    Eigen::MatrixXi tris;
  } beam3d;

  VulkanExample() : VulkanExampleBase() {
    title = "Compute shader deformable simulation";
    camera.type = Camera::CameraType::lookat;
    camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
    camera.setRotation(glm::vec3(-30.0f, -45.0f, 0.0f));
    camera.setTranslation(glm::vec3(0.0f, -0.0f, -8.0f));
  }

  ~VulkanExample() {
    if (device) {
      // Graphics
      graphics.indices.destroy();
      for (auto &buffer : graphics.uniformBuffers) {
        buffer.destroy();
      }
      vkDestroyPipeline(device, graphics.pipelines.beam3d, nullptr);
      vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout,
                                   nullptr);

      // Compute
      compute.uniformBuffer.destroy();
      vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout,
                                   nullptr);
      vkDestroyPipeline(device, compute.pipelines.begin, nullptr);
      vkDestroyPipeline(device, compute.pipelines.solve, nullptr);
      vkDestroyPipeline(device, compute.pipelines.end, nullptr);
      for (auto &fence : compute.fences) {
        vkDestroyFence(device, fence, nullptr);
      }
      for (auto &semaphore : compute.semaphores) {
        vkDestroySemaphore(device, semaphore.ready, nullptr);
        vkDestroySemaphore(device, semaphore.complete, nullptr);
      }
      vkDestroyCommandPool(device, compute.commandPool, nullptr);

      // SSBOs
      storageBuffers.particles.destroy();
      if (outputReadback.buffer != VK_NULL_HANDLE) {
        outputReadback.unmap();
        outputReadback.destroy();
      }
    }
  }

  // Setup and fill the shader storage buffers containing the particles
  // These buffers are used as shader storage buffers in the compute shader (to
  // update them) and as vertex input in the vertex shader (to display them)
  void prepareStorageBuffers() {
    uint32_t numParticles = static_cast<uint32_t>(beam3d.V.rows());

    std::vector<Particle> particleBuffer(numParticles);
    for (uint32_t i = 0; i < numParticles; i++) {
      particleBuffer[i].pos =
          glm::vec4(beam3d.V(i, 0), beam3d.V(i, 1), beam3d.V(i, 2), 1.0f);
      particleBuffer[i].vel = glm::vec4(0.0f);
      particleBuffer[i].uv = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
      particleBuffer[i].normal = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int i = 0; i < beam3d.tris.rows(); i++) {
      int a = beam3d.tris(i, 0);
      int b = beam3d.tris(i, 1);
      int c = beam3d.tris(i, 2);
      glm::vec3 normal = glm::cross(
          glm::vec3(particleBuffer[b].pos) - glm::vec3(particleBuffer[a].pos),
          glm::vec3(particleBuffer[c].pos) - glm::vec3(particleBuffer[a].pos));
      particleBuffer[a].normal += glm::vec4(normal, 0.0f);
      particleBuffer[b].normal += glm::vec4(normal, 0.0f);
      particleBuffer[c].normal += glm::vec4(normal, 0.0f);
    }
    for (int i = 0; i < numParticles; i++) {
      particleBuffer[i].normal = glm::normalize(particleBuffer[i].normal);
    }

    storageBufferSize = particleBuffer.size() * sizeof(Particle);

    vks::Buffer stagingBuffer;

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, storageBufferSize,
                               particleBuffer.data());

    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &storageBuffers.particles,
        storageBufferSize);

    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = storageBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer,
                    storageBuffers.particles.buffer, 1, &copyRegion);
    example_barriers::addGraphicsToComputeBarriers(
        copyCmd, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
        vulkanDevice->queueFamilyIndices.graphics,
        vulkanDevice->queueFamilyIndices.compute, VK_ACCESS_TRANSFER_WRITE_BIT,
        0, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();

    if (writeVtk) {
      // Host-visible buffer used only when particle states are exported to VTK.
      vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &outputReadback, storageBufferSize);
      VK_CHECK_RESULT(outputReadback.map());
    }

    // Index buffer from model indices (triangle list)
    uint32_t numSurfacePoints =
        static_cast<uint32_t>(beam3d.tris.rows() * beam3d.tris.cols());

    uint32_t indexBufferSize =
        static_cast<uint32_t>(numSurfacePoints) * sizeof(uint32_t);
    indexCount = static_cast<uint32_t>(numSurfacePoints);

    std::vector<uint32_t> indexPointBuffer(numSurfacePoints);
    for (uint32_t i = 0; i < numSurfacePoints; i++) {
      indexPointBuffer[i] = beam3d.tris(i / 3, i % 3);
    }

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, indexBufferSize,
                               indexPointBuffer.data());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &graphics.indices, indexBufferSize);

    copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                true);
    copyRegion = {};
    copyRegion.size = indexBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, graphics.indices.buffer, 1,
                    &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();
    uint32_t numElements = static_cast<uint32_t>(compute.elementInfo.size());
    compute.lambdaDData.resize(numElements, 0.0f);
    compute.lambdaHData.resize(numElements, 0.0f);
    // ElementInfo buffer
    VkDeviceSize elementInfoBufferSize = numElements * sizeof(ElementInfo);
    vks::Buffer stagingElementInfoBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingElementInfoBuffer, elementInfoBufferSize,
                               compute.elementInfo.data());
    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &compute.elementInfoBuffer,
        elementInfoBufferSize);
    // LambdaD buffer
    const VkDeviceSize lambdaBufferSize = numElements * sizeof(float);
    vks::Buffer stagingLambdaBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingLambdaBuffer, lambdaBufferSize,
                               compute.lambdaDData.data());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &compute.lambdaDBuffer, lambdaBufferSize);
    // LambdaH buffer
    vks::Buffer stagingLambdaHBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingLambdaHBuffer, lambdaBufferSize,
                               compute.lambdaHData.data());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &compute.lambdaHBuffer, lambdaBufferSize);
    // mass buffer
    vks::Buffer stagingMassesBuffer;
    VkDeviceSize massesBufferSize = compute.masses.size() * sizeof(float);
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingMassesBuffer, massesBufferSize,
                               compute.masses.data());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &compute.massesBuffer, massesBufferSize);
    vks::Buffer stagingfixedpointBuffer;
    VkDeviceSize fixedpointBufferSize = compute.fixedpoint.size() * sizeof(int);
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingfixedpointBuffer, fixedpointBufferSize,
                               compute.fixedpoint.data());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &compute.fixedpointBuffer, fixedpointBufferSize);

    //  ElemParallelSlots buffer
    uint32_t numSlots = static_cast<uint32_t>(compute.elemParallelSlots.size());
    VkDeviceSize elemParallelSlotsBufferSize = numSlots * sizeof(int);
    vks::Buffer stagingElemParallelSlotsBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingElemParallelSlotsBuffer,
                               elemParallelSlotsBufferSize,
                               compute.elemParallelSlots.data());
    vulkanDevice->createBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &compute.elemParallelSlotsBuffer,
        elemParallelSlotsBufferSize);
    copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                true);
    copyRegion = {};
    copyRegion.size = elementInfoBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElementInfoBuffer.buffer,
                    compute.elementInfoBuffer.buffer, 1, &copyRegion);
    copyRegion.size = lambdaBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingLambdaBuffer.buffer,
                    compute.lambdaDBuffer.buffer, 1, &copyRegion);
    vkCmdCopyBuffer(copyCmd, stagingLambdaHBuffer.buffer,
                    compute.lambdaHBuffer.buffer, 1, &copyRegion);
    copyRegion.size = massesBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingMassesBuffer.buffer,
                    compute.massesBuffer.buffer, 1, &copyRegion);
    copyRegion.size = fixedpointBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingfixedpointBuffer.buffer,
                    compute.fixedpointBuffer.buffer, 1, &copyRegion);
    copyRegion.size = elemParallelSlotsBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElemParallelSlotsBuffer.buffer,
                    compute.elemParallelSlotsBuffer.buffer, 1, &copyRegion);

    // These buffers are initialized on the graphics queue but used only by
    // compute. Release their ownership here; the first compute command buffer
    // acquires them before binding the compute pipeline.
    example_barriers::addGraphicsToComputeBarriers(
        copyCmd,
        {compute.elementInfoBuffer.buffer, compute.lambdaDBuffer.buffer,
         compute.lambdaHBuffer.buffer, compute.massesBuffer.buffer,
         compute.fixedpointBuffer.buffer,
         compute.elemParallelSlotsBuffer.buffer},
        dedicatedComputeQueue, vulkanDevice->queueFamilyIndices.graphics,
        vulkanDevice->queueFamilyIndices.compute, VK_ACCESS_TRANSFER_WRITE_BIT,
        0, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
    // Clean up staging buffers
    stagingElementInfoBuffer.destroy();
    stagingLambdaBuffer.destroy();
    stagingElemParallelSlotsBuffer.destroy();
    stagingLambdaHBuffer.destroy();
    stagingMassesBuffer.destroy();
    stagingfixedpointBuffer.destroy();
  }

  void prepareDescriptorPool() {
    // This is shared between graphics and compute
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 3),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              maxConcurrentFrames * 12)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    maxConcurrentFrames * 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                           &descriptorPool));
  }

  // Prepare the resources used for the graphics part of the sample
  void prepareGraphics() {
    // Uniform buffers for passing data to the vertex shader
    for (auto &buffer : graphics.uniformBuffers) {
      vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 &buffer, sizeof(Graphics::UniformData));
      VK_CHECK_RESULT(buffer.map());
    }

    // Descriptor layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0)};
    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

    // Sets per frame in flight as the uniform buffer is written by the CPU and
    // read by the GPU
    for (auto i = 0; i < graphics.uniformBuffers.size(); i++) {
      VkDescriptorSetAllocateInfo allocInfo =
          vks::initializers::descriptorSetAllocateInfo(
              descriptorPool, &graphics.descriptorSetLayout, 1);
      VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                               &graphics.descriptorSets[i]));
      std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
          vks::initializers::writeDescriptorSet(
              graphics.descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0,
              &graphics.uniformBuffers[i].descriptor)};
      vkUpdateDescriptorSets(device,
                             static_cast<uint32_t>(writeDescriptorSets.size()),
                             writeDescriptorSets.data(), 0, nullptr);
    }

    // Layout
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &graphics.descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr, &graphics.pipelineLayout));

    // Pipeline
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
        vks::initializers::pipelineInputAssemblyStateCreateInfo(
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
    VkPipelineRasterizationStateCreateInfo rasterizationState =
        vks::initializers::pipelineRasterizationStateCreateInfo(
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
    VkPipelineColorBlendAttachmentState blendAttachmentState =
        vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
    VkPipelineColorBlendStateCreateInfo colorBlendState =
        vks::initializers::pipelineColorBlendStateCreateInfo(
            1, &blendAttachmentState);
    VkPipelineDepthStencilStateCreateInfo depthStencilState =
        vks::initializers::pipelineDepthStencilStateCreateInfo(
            VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
    VkPipelineViewportStateCreateInfo viewportState =
        vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
    VkPipelineMultisampleStateCreateInfo multisampleState =
        vks::initializers::pipelineMultisampleStateCreateInfo(
            VK_SAMPLE_COUNT_1_BIT, 0);
    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState =
        vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

    // Rendering pipeline
    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        loadShader(getShadersPath() + "riddfmb3d/beam3d.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "riddfmb3d/beam3d.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT)};

    VkGraphicsPipelineCreateInfo pipelineCreateInfo =
        vks::initializers::pipelineCreateInfo(graphics.pipelineLayout,
                                              renderPass);

    // Vertex Input
    std::vector<VkVertexInputBindingDescription> inputBindings = {
        vks::initializers::vertexInputBindingDescription(
            0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)};
    // Attribute descriptions based on the particles of the cloth
    std::vector<VkVertexInputAttributeDescription> inputAttributes = {
        vks::initializers::vertexInputAttributeDescription(
            0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, pos)),
        vks::initializers::vertexInputAttributeDescription(
            0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(Particle, uv)),
        vks::initializers::vertexInputAttributeDescription(
            0, 2, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, normal))};

    // Assign to vertex buffer
    VkPipelineVertexInputStateCreateInfo inputState =
        vks::initializers::pipelineVertexInputStateCreateInfo();
    inputState.vertexBindingDescriptionCount =
        static_cast<uint32_t>(inputBindings.size());
    inputState.pVertexBindingDescriptions = inputBindings.data();
    inputState.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(inputAttributes.size());
    inputState.pVertexAttributeDescriptions = inputAttributes.data();

    pipelineCreateInfo.pVertexInputState = &inputState;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.renderPass = renderPass;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics.pipelines.beam3d));
  }

  void Boundarycondition() {
    uint32_t numParticles = static_cast<uint32_t>(beam3d.V.rows());
    compute.fixedpoint.resize(numParticles, 0);
  }

  void buildElementInfoFromMesh() {
#if (_android_)
    io::readMesh3d(config.modelPath, beam3d.V, beam3d.tets);
#else
    io::readMesh3d(getAssetPath() + config.modelPath, beam3d.V, beam3d.tets);
#endif
    io::internal::extractBoundary(beam3d.V, beam3d.tets, beam3d.tris);
    compute.elementInfo.clear();
    for (size_t i = 0; i < beam3d.tets.rows(); i++) {
      ElementInfo elem{};
      elem.elemId = static_cast<int>(compute.elementInfo.size());
      elem.pid = glm::ivec4(beam3d.tets(i, 0), beam3d.tets(i, 1),
                            beam3d.tets(i, 2), beam3d.tets(i, 3));
      glm::vec3 p0(beam3d.V(beam3d.tets(i, 0), 0),
                   beam3d.V(beam3d.tets(i, 0), 1),
                   beam3d.V(beam3d.tets(i, 0), 2));
      glm::vec3 p1(beam3d.V(beam3d.tets(i, 1), 0),
                   beam3d.V(beam3d.tets(i, 1), 1),
                   beam3d.V(beam3d.tets(i, 1), 2));
      glm::vec3 p2(beam3d.V(beam3d.tets(i, 2), 0),
                   beam3d.V(beam3d.tets(i, 2), 1),
                   beam3d.V(beam3d.tets(i, 2), 2));
      glm::vec3 p3(beam3d.V(beam3d.tets(i, 3), 0),
                   beam3d.V(beam3d.tets(i, 3), 1),
                   beam3d.V(beam3d.tets(i, 3), 2));
      glm::vec3 e1 = p1 - p0;
      glm::vec3 e2 = p2 - p0;
      glm::vec3 e3 = p3 - p0;
      elem.restShape = glm::mat3x4(glm::vec4(e1, 0.0f), glm::vec4(e2, 0.0f),
                                   glm::vec4(e3, 0.0f));
      glm::mat3 temp_restShape = glm::mat3(e1, e2, e3);
      elem.restShapeInv = glm::inverse(temp_restShape);
      elem.restVol = std::abs(glm::determinant(temp_restShape)) / 6.0f;
      compute.elementInfo.push_back(elem);
    }
  }
  void precompute() {
    uint32_t nelements = compute.elementInfo.size();
    uint32_t nparticles = beam3d.V.rows();
    // Step-1: precompute element parallelable sets
    std::vector<std::vector<int>> elemParaSets;
    {
      std::vector<int> elementIds;
      elementIds.reserve(compute.elementInfo.size());
      for (int i = 0; i < nelements; ++i) {
        elementIds.emplace_back(i);
      }
      while (!elementIds.empty()) {
        std::vector<bool> particleOccupied(nparticles, false);
        std::vector<int> currentSet;
        for (auto it = elementIds.begin(); it != elementIds.end();) {
          const auto &elemInfo = compute.elementInfo[*it];
          bool canAdd = true;
          for (int i = 0; i < 4; ++i) {
            if (particleOccupied[elemInfo.pid[i]]) {
              canAdd = false;
              break;
            }
          }
          if (canAdd) {
            // add to current set
            currentSet.emplace_back(*it);
            for (int i = 0; i <= 3; ++i) {
              const auto pid = elemInfo.pid[i];
              particleOccupied[pid] = true;
            }

            // remove from elementIds
            it = elementIds.erase(it);
          } else {
            ++it;
          }
        }
        elemParaSets.emplace_back(std::move(currentSet));
      }
      // reorder elemInfos according to parallelable sets
      std::vector<ElementInfo> reorderedElemInfos;
      reorderedElemInfos.reserve(nelements);
      compute.elemParallelSlots.clear();
      for (const auto &elemIdSet : elemParaSets) {
        compute.elemParallelSlots.emplace_back(
            static_cast<int>(reorderedElemInfos.size()));
        for (const auto elemId : elemIdSet) {
          reorderedElemInfos.emplace_back(compute.elementInfo[elemId]);
          // correct elemId
          reorderedElemInfos.back().elemId =
              static_cast<int>(reorderedElemInfos.size()) - 1;
        }
      }
      compute.elemParallelSlots.emplace_back(
          static_cast<int>(reorderedElemInfos.size()));
      assert(reorderedElemInfos.size() == nelements);
      std::swap(compute.elementInfo, reorderedElemInfos);
    }

    // Step-2: Precompute rest Shape Matrix, rest Volume, mass and inverse
    {
      compute.masses.resize(nparticles);
      std::fill(compute.masses.begin(), compute.masses.end(), float(0.0));
      for (int i = 0; i < nelements; ++i) {
        const auto &info = compute.elementInfo[i];
        const float mass = compute.uniformData.density *
                           info.restVol; // mass = density * volume
        for (int j = 0; j <= 3; ++j) {
          compute.masses[info.pid[j]] += mass / static_cast<float>(4.0);
        }
      }
    }
  }

  // Prepare the resources used for the compute part of the sample
  void prepareCompute() {
    // Create a compute capable device queue
    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.compute, 0,
                     &compute.queue);

    // Uniform buffer for passing data to the compute shader
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &compute.uniformBuffer,
                               sizeof(Compute::UniformData));
    VK_CHECK_RESULT(compute.uniformBuffer.map());

    // Create compute pipeline
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 7),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 8),
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute.descriptorSetLayout, 1);

    // Push constant used for the solve stage (parallel set index)
    VkPushConstantRange pushConstantRange =
        vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT,
                                             sizeof(PushConstants), 0);
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo,
                                           nullptr, &compute.pipelineLayout));

    VkDescriptorSetAllocateInfo allocInfo =
        vks::initializers::descriptorSetAllocateInfo(
            descriptorPool, &compute.descriptorSetLayout, 1);

    // Single descriptor set, fixed binding of input/output buffers
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                             &compute.descriptorSets[0]));

    std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0,
            &storageBuffers.particles.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
            &compute.uniformBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3,
            &compute.lambdaDBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
            &compute.elementInfoBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5,
            &compute.elemParallelSlotsBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6,
            &compute.massesBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7,
            &compute.lambdaHBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8,
            &compute.fixedpointBuffer.descriptor)};

    vkUpdateDescriptorSets(
        device, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
        computeWriteDescriptorSets.data(), 0, NULL);

    // Create pipelines (begin / solve / end)
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb3d/cloth_begin.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.begin));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb3d/cloth_solve.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.solve));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb3d/cloth_end.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1,
                                             &computePipelineCreateInfo,
                                             nullptr, &compute.pipelines.end));

    // Separate command pool as queue family for compute may be different than
    // graphics
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr,
                                        &compute.commandPool));

    // Create command buffers for compute operations
    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
        vks::initializers::commandBufferAllocateInfo(
            compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            static_cast<uint32_t>(compute.commandBuffers.size()));
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo,
                                             &compute.commandBuffers[0]));

    // Fences to check for command buffer completion
    for (auto &fence : compute.fences) {
      VkFenceCreateInfo fenceCreateInfo =
          vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
      VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
    }

    // Semaphores for graphics / compute synchronization
    VkSemaphoreCreateInfo semaphoreCreateInfo =
        vks::initializers::semaphoreCreateInfo();
    for (uint32_t i = 0; i < compute.semaphores.size(); i++) {
      VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr,
                                        &compute.semaphores[i].ready));
      VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr,
                                        &compute.semaphores[i].complete));
    }
    // Signal first used ready semaphore
    VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
    computeSubmitInfo.signalSemaphoreCount = 1;
    computeSubmitInfo.pSignalSemaphores =
        &compute.semaphores[maxConcurrentFrames - 1].ready;
    VK_CHECK_RESULT(
        vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));
  }

  void updateComputeUBO() {
    if (!paused) {
      compute.uniformData.deltaT = config.deltaT;
    } else {
      compute.uniformData.deltaT = 0.0f;
    }
    memcpy(compute.uniformBuffer.mapped, &compute.uniformData,
           sizeof(Compute::UniformData));
  }

  void applyConfigurationToCompute() {
    compute.uniformData.density = config.density;
    compute.uniformData.gravity = config.gravity;
    compute.uniformData.lame = config.lame;
  }

  void updateGraphicsUBO() {
    graphics.uniformData.projection = camera.matrices.perspective;
    const glm::mat4 flipY =
        glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, 1.0f));
    // Flip model Y axis (model = flipY, modelview = view * model)
    graphics.uniformData.view = camera.matrices.view * flipY;
    memcpy(graphics.uniformBuffers[currentBuffer].mapped, &graphics.uniformData,
           sizeof(Graphics::UniformData));
  }
  void prepare() {
    VulkanExampleBase::prepare();
    example_config::loadRiddfmb3dConfiguration(args, config);
    writeVtk = example_config::hasCommandLineFlag(args, "--write-vtk");
    if (writeVtk) {
      std::cout << "riddfmb3d: VTK export enabled\n";
    }
    applyConfigurationToCompute();
    // Check whether the compute queue family is distinct from the graphics
    // queue family
    dedicatedComputeQueue = vulkanDevice->queueFamilyIndices.graphics !=
                            vulkanDevice->queueFamilyIndices.compute;
    std::cout << dedicatedComputeQueue << std::endl;
    buildElementInfoFromMesh();
    precompute();
    Boundarycondition();
    prepareStorageBuffers();
    prepareDescriptorPool();
    prepareGraphics();
    prepareCompute();
    prepared = true;
  }

  void buildGraphicsCommandBuffer() {
    VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2]{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo =
        vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.framebuffer = frameBuffers[currentImageIndex];

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Acquire storage buffers from compute queue
    example_barriers::addComputeToGraphicsBarriers(
        cmdBuffer, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
        vulkanDevice->queueFamilyIndices.compute,
        vulkanDevice->queueFamilyIndices.graphics, 0,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

    // Draw the particle system using the update vertex buffer

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport =
        vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    VkDeviceSize offsets[1] = {0};

    // Render cloth
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics.pipelines.beam3d);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0,
        1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
    vkCmdBindIndexBuffer(cmdBuffer, graphics.indices.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &storageBuffers.particles.buffer,
                           offsets);
    vkCmdDrawIndexed(cmdBuffer, indexCount, 1, 0, 0, 0);

    drawUI(cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

    // release the storage buffers to the compute queue
    example_barriers::addGraphicsToComputeBarriers(
        cmdBuffer, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
        vulkanDevice->queueFamilyIndices.graphics,
        vulkanDevice->queueFamilyIndices.compute,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 0,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void buildComputeCommandBuffer(uint32_t substeps) {
    VkCommandBuffer cmdBuffer = compute.commandBuffers[currentBuffer];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Acquire the storage buffers from the graphics queue
    example_barriers::addGraphicsToComputeBarriers(
        cmdBuffer, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
        vulkanDevice->queueFamilyIndices.graphics,
        vulkanDevice->queueFamilyIndices.compute, 0,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    if (!computeOnlyBuffersAcquired) {
      // Match the release after the initialization copies. These buffers remain
      // owned by the compute queue for the lifetime of the example.
      example_barriers::addGraphicsToComputeBarriers(
          cmdBuffer,
          {compute.elementInfoBuffer.buffer, compute.lambdaDBuffer.buffer,
           compute.lambdaHBuffer.buffer, compute.massesBuffer.buffer,
           compute.fixedpointBuffer.buffer,
           compute.elemParallelSlotsBuffer.buffer},
          dedicatedComputeQueue, vulkanDevice->queueFamilyIndices.graphics,
          vulkanDevice->queueFamilyIndices.compute, 0,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      computeOnlyBuffersAcquired = true;
    }

    // If we don't need to advance simulation this frame, we still need to
    // transfer ownership back to graphics (graphics command buffer will
    // acquire).
    if (substeps == 0) {
      example_barriers::addComputeToGraphicsBarriers(
          cmdBuffer, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
          vulkanDevice->queueFamilyIndices.compute,
          vulkanDevice->queueFamilyIndices.graphics, 0, 0,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      vkEndCommandBuffer(cmdBuffer);
      return;
    }

    // Single descriptor set, fixed binding of input/output buffers
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[0], 0, 0);

    const uint32_t numParticles = static_cast<uint32_t>(beam3d.V.rows());
    const uint32_t numLambda =
        static_cast<uint32_t>(compute.lambdaDData.size());
    const uint32_t workgroupSizeX = 64;
    const uint32_t numWorkgroupsX =
        (std::max(numParticles, numLambda) + workgroupSizeX - 1) /
        workgroupSizeX;

    const uint32_t numParallelSets =
        static_cast<uint32_t>(compute.elemParallelSlots.size()) - 1;
    const uint32_t constraintIterations = config.numSolverIterations;

    const uint32_t workgroupSizeXSet = 64;
    const uint32_t numParticlesStage2 = numParticles;
    const uint32_t workgroupSizeXStage2 = 64;
    const uint32_t numWorkgroupsXStage2 =
        (numParticlesStage2 + workgroupSizeXStage2 - 1) / workgroupSizeXStage2;

    PushConstants pushConsts{};
    for (uint32_t step = 0; step < substeps; step++) {
      // Stage 0: Begin solve
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.begin);
      vkCmdDispatch(cmdBuffer, numWorkgroupsX, 1, 1);

      // Barrier after begin solve
      example_barriers::addComputeToComputeBarriers(
          cmdBuffer,
          {storageBuffers.particles.buffer, compute.lambdaDBuffer.buffer});

      // Stage 1: Constraint solving
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.solve);

      for (uint32_t iter = 0; iter < constraintIterations; iter++) {
        for (uint32_t setIdx = 0; setIdx < numParallelSets; setIdx++) {
          pushConsts.parallelSetStartIndex = setIdx;
          vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(PushConstants), &pushConsts);

          const uint32_t setStart = compute.elemParallelSlots[setIdx];
          const uint32_t setEnd = compute.elemParallelSlots[setIdx + 1];
          const uint32_t setSize = setEnd - setStart;

          const uint32_t numWorkgroupsXSet =
              (setSize + workgroupSizeXSet - 1) / workgroupSizeXSet;
          vkCmdDispatch(cmdBuffer, numWorkgroupsXSet, 1, 1);

          if (setIdx < numParallelSets - 1) {
            example_barriers::addComputeToComputeBarriers(
                cmdBuffer, {storageBuffers.particles.buffer,
                            compute.lambdaDBuffer.buffer});
          }
        }
        if (iter < constraintIterations - 1) {
          example_barriers::addComputeToComputeBarriers(
              cmdBuffer,
              {storageBuffers.particles.buffer, compute.lambdaDBuffer.buffer});
        }
      }

      // Stage 2: End solve
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.end);
      vkCmdDispatch(cmdBuffer, numWorkgroupsXStage2, 1, 1);

      // Barrier between substeps (except after the last one)
      if (step + 1 < substeps) {
        example_barriers::addComputeToComputeBarriers(
            cmdBuffer,
            {storageBuffers.particles.buffer, compute.lambdaDBuffer.buffer});
      }
    }

    if (writeVtk) {
      // Copy the completed particle state for CPU-side VTK export.
      VkBufferMemoryBarrier copyBarrier =
          vks::initializers::bufferMemoryBarrier();
      copyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      copyBarrier.size = VK_WHOLE_SIZE;
      copyBarrier.buffer = storageBuffers.particles.buffer;
      vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_FLAGS_NONE, 0,
                           nullptr, 1, &copyBarrier, 0, nullptr);

      VkBufferCopy copyRegion = {};
      copyRegion.size = storageBufferSize;
      vkCmdCopyBuffer(cmdBuffer, storageBuffers.particles.buffer,
                      outputReadback.buffer, 1, &copyRegion);
    }

    // Release the storage buffers back to the graphics queue
    example_barriers::addComputeToGraphicsBarriers(
        cmdBuffer, {storageBuffers.particles.buffer}, dedicatedComputeQueue,
        vulkanDevice->queueFamilyIndices.compute,
        vulkanDevice->queueFamilyIndices.graphics,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, 0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmdBuffer);
  }

  virtual void render() {
    if (!prepared)
      return;
    // Submit compute commands
    // auto startTime = std::chrono::high_resolution_clock::now();
    {
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));
      VK_CHECK_RESULT(vkResetFences(device, 1, &compute.fences[currentBuffer]));
      updateComputeUBO();
      uint32_t substeps =
          static_cast<uint32_t>(std::floor(1 / (FPS * config.deltaT)));
      buildComputeCommandBuffer(substeps);

      VkPipelineStageFlags waitDstStageMask =
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      VkSubmitInfo submitInfo = vks::initializers::submitInfo();
      submitInfo.waitSemaphoreCount = 1;
      submitInfo.pWaitSemaphores =
          &compute.semaphores[((int)currentBuffer - 1) % maxConcurrentFrames]
               .ready;
      submitInfo.pWaitDstStageMask = &waitDstStageMask;
      submitInfo.signalSemaphoreCount = 1;
      submitInfo.pSignalSemaphores =
          &compute.semaphores[currentBuffer].complete;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &compute.commandBuffers[currentBuffer];
      VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &submitInfo,
                                    compute.fences[currentBuffer]));
      if (writeVtk && substeps > 0) {
        readbackPending = true;
        readbackFenceIndex = currentBuffer;
        readbackFrame = Framecount;
      }
    }

    //  Submit graphics commands
    {
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));
      VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));

      VulkanExampleBase::prepareFrame(false);

      updateGraphicsUBO();
      buildGraphicsCommandBuffer();

      VkPipelineStageFlags waitDstStageMask[2] = {
          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
      VkSemaphore waitSemaphores[2] = {
          presentCompleteSemaphores[currentBuffer],
          compute.semaphores[currentBuffer].complete};
      VkSemaphore signalSemaphores[2] = {
          renderCompleteSemaphores[currentImageIndex],
          compute.semaphores[currentBuffer].ready};

      VkSubmitInfo submitInfo = vks::initializers::submitInfo();
      submitInfo.waitSemaphoreCount = 2;
      submitInfo.pWaitSemaphores = waitSemaphores;
      submitInfo.pWaitDstStageMask = waitDstStageMask;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
      submitInfo.signalSemaphoreCount = 2;
      submitInfo.pSignalSemaphores = signalSemaphores;
      VK_CHECK_RESULT(
          vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]));

      VulkanExampleBase::submitFrame(true);
    }
    Framecount++;
  }

  virtual void writeMesh() {
    if (!writeVtk || !readbackPending) {
      return;
    }

    VK_CHECK_RESULT(vkWaitForFences(
        device, 1, &compute.fences[readbackFenceIndex], VK_TRUE, UINT64_MAX));
    const auto *particles =
        reinterpret_cast<const Particle *>(outputReadback.mapped);
    for (int i = 0; i < beam3d.V.rows(); i++) {
      beam3d.V(i, 0) = particles[i].pos.x;
      beam3d.V(i, 1) = particles[i].pos.y;
      beam3d.V(i, 2) = particles[i].pos.z;
    }

    const std::filesystem::path outputDirectory{"../output"};
    std::filesystem::create_directories(outputDirectory);
    const std::filesystem::path outputPath =
        outputDirectory / ("RIDdfmb_" + std::to_string(readbackFrame) + ".vtk");
    VtkOutput vtkoutput(outputPath);
    vtkoutput.writeMesh<VtkCellType::TETRA>(beam3d.V, beam3d.tets);
    readbackPending = false;
  }
};

VULKAN_EXAMPLE_MAIN()
