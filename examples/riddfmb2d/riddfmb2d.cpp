
#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"
#include <cstdint>
#include <cstdio>

class VulkanExample : public VulkanExampleBase {
public:
  uint32_t indexCount{0};
  bool dedicatedComputeQueue{false};

  vkglTF::Model modelBeam2d;

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
    alignas(16) glm::mat2x2 restShape;
    alignas(16) glm::mat2x2 restShapeInv;
  };

  struct StorageBuffers {
    vks::Buffer input;
    vks::Buffer output;
  } storageBuffers;

  struct PushConstants {
    uint32_t parallelSetStartIndex;
  };

  struct Graphics {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline beam2d{VK_NULL_HANDLE};
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
      float density{100.0f};
      alignas(16) glm::vec4 gravity{0.0f, 9.8f, 0.0f, 0.0f};
      glm::vec4 lame{100000.0f, 100000.0f, 0.0f, 0.0f};
      glm::ivec2 particleCount{0};
    } uniformData;
    std::vector<float> masses{};
    std::vector<ElementInfo> elementInfo;
    // RID solves one scalar multiplier per element.
    std::vector<float> lambdaData;
    std::vector<int> elemParallelSlots;
    vks::Buffer uniformBuffer;
    vks::Buffer lambdaBuffer;
    vks::Buffer elementInfoBuffer;
    vks::Buffer elemParallelSlotsBuffer;
    vks::Buffer massesBuffer;
    std::vector<int> fixedpoint;
    vks::Buffer fixedpointBuffer;
  } compute;

  VulkanExample() : VulkanExampleBase() {
    title = "Compute shader deformable simulation";
    camera.type = Camera::CameraType::lookat;
    camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
    camera.setRotation(glm::vec3(-30.0f, -45.0f, 0.0f));
    camera.setTranslation(glm::vec3(0.0f, 0.0f, -5.0f));
  }

  ~VulkanExample() {
    if (device) {
      // Graphics
      graphics.indices.destroy();
      for (auto &buffer : graphics.uniformBuffers) {
        buffer.destroy();
      }
      vkDestroyPipeline(device, graphics.pipelines.beam2d, nullptr);
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
      storageBuffers.input.destroy();
      storageBuffers.output.destroy();
    }
  }

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    if (deviceFeatures.samplerAnisotropy) {
      enabledFeatures.samplerAnisotropy = VK_TRUE;
    }
  };

  void loadAssets() {
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    modelBeam2d.loadFromFile(getAssetPath() + "models/beam.gltf", vulkanDevice,
                             queue, glTFLoadingFlags);
  }

  void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer,
                                    VkAccessFlags srcAccessMask,
                                    VkAccessFlags dstAccessMask,
                                    VkPipelineStageFlags srcStageMask,
                                    VkPipelineStageFlags dstStageMask) {
    if (dedicatedComputeQueue) {
      VkBufferMemoryBarrier bufferBarrier =
          vks::initializers::bufferMemoryBarrier();
      bufferBarrier.srcAccessMask = srcAccessMask;
      bufferBarrier.dstAccessMask = dstAccessMask;
      bufferBarrier.srcQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.graphics;
      bufferBarrier.dstQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.compute;
      bufferBarrier.size = VK_WHOLE_SIZE;

      std::vector<VkBufferMemoryBarrier> bufferBarriers;
      bufferBarrier.buffer = storageBuffers.input.buffer;
      bufferBarriers.push_back(bufferBarrier);
      bufferBarrier.buffer = storageBuffers.output.buffer;
      bufferBarriers.push_back(bufferBarrier);
      vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask,
                           VK_FLAGS_NONE, 0, nullptr,
                           static_cast<uint32_t>(bufferBarriers.size()),
                           bufferBarriers.data(), 0, nullptr);
    }
  }

  void addComputeToComputeBarriers(VkCommandBuffer commandBuffer) {
    VkBufferMemoryBarrier bufferBarrier =
        vks::initializers::bufferMemoryBarrier();
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.size = VK_WHOLE_SIZE;
    std::vector<VkBufferMemoryBarrier> bufferBarriers;
    // Conservatively add a memory barrier for both particle buffers
    bufferBarrier.buffer = storageBuffers.input.buffer;
    bufferBarriers.push_back(bufferBarrier);
    bufferBarrier.buffer = storageBuffers.output.buffer;
    bufferBarriers.push_back(bufferBarrier);
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_FLAGS_NONE, 0,
                         nullptr, static_cast<uint32_t>(bufferBarriers.size()),
                         bufferBarriers.data(), 0, nullptr);
  }

  void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer,
                                    VkAccessFlags srcAccessMask,
                                    VkAccessFlags dstAccessMask,
                                    VkPipelineStageFlags srcStageMask,
                                    VkPipelineStageFlags dstStageMask) {
    if (dedicatedComputeQueue) {
      VkBufferMemoryBarrier bufferBarrier =
          vks::initializers::bufferMemoryBarrier();
      bufferBarrier.srcAccessMask = srcAccessMask;
      bufferBarrier.dstAccessMask = dstAccessMask;
      bufferBarrier.srcQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.compute;
      bufferBarrier.dstQueueFamilyIndex =
          vulkanDevice->queueFamilyIndices.graphics;
      bufferBarrier.size = VK_WHOLE_SIZE;
      std::vector<VkBufferMemoryBarrier> bufferBarriers;
      bufferBarrier.buffer = storageBuffers.input.buffer;
      bufferBarriers.push_back(bufferBarrier);
      bufferBarrier.buffer = storageBuffers.output.buffer;
      bufferBarriers.push_back(bufferBarrier);
      vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask,
                           VK_FLAGS_NONE, 0, nullptr,
                           static_cast<uint32_t>(bufferBarriers.size()),
                           bufferBarriers.data(), 0, nullptr);
    }
  }

  // Setup and fill the shader storage buffers containing the particles
  // These buffers are used as shader storage buffers in the compute shader (to
  // update them) and as vertex input in the vertex shader (to display them)
  void prepareStorageBuffers() {
    auto &verts = modelBeam2d.cpuVertices;
    auto &inds = modelBeam2d.cpuIndices;
    uint32_t numParticles = static_cast<uint32_t>(verts.size());

    std::vector<Particle> particleBuffer(numParticles);
    for (uint32_t i = 0; i < numParticles; i++) {
      particleBuffer[i].pos = glm::vec4(verts[i].pos, 1.0f);
      particleBuffer[i].vel = glm::vec4(0.0f);
      particleBuffer[i].uv = glm::vec4(verts[i].uv, 0.0f, 0.0f);
      particleBuffer[i].normal = glm::vec4(verts[i].normal, 0.0f);
    }

    VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

    vks::Buffer stagingBuffer;

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, storageBufferSize,
                               particleBuffer.data());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &storageBuffers.input, storageBufferSize);

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &storageBuffers.output, storageBufferSize);

    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = storageBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.output.buffer,
                    1, &copyRegion);
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.input.buffer,
                    1, &copyRegion);
    addGraphicsToComputeBarriers(copyCmd, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 0,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();

    // Index buffer from model indices (triangle list)
    uint32_t indexBufferSize =
        static_cast<uint32_t>(inds.size()) * sizeof(uint32_t);
    indexCount = static_cast<uint32_t>(inds.size());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, indexBufferSize, inds.data());

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
    compute.lambdaData.resize(numElements, 0.0f);
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
    // RID multiplier buffer
    VkDeviceSize lambdaBufferSize = numElements * sizeof(float);
    vks::Buffer stagingLambdaBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingLambdaBuffer, lambdaBufferSize,
                               compute.lambdaData.data());
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &compute.lambdaBuffer, lambdaBufferSize);
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
                    compute.lambdaBuffer.buffer, 1, &copyRegion);
    copyRegion.size = massesBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingMassesBuffer.buffer,
                    compute.massesBuffer.buffer, 1, &copyRegion);
    copyRegion.size = fixedpointBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingfixedpointBuffer.buffer,
                    compute.fixedpointBuffer.buffer, 1, &copyRegion);
    copyRegion.size = elemParallelSlotsBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElemParallelSlotsBuffer.buffer,
                    compute.elemParallelSlotsBuffer.buffer, 1, &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
    // Clean up staging buffers
    stagingElementInfoBuffer.destroy();
    stagingLambdaBuffer.destroy();
    stagingElemParallelSlotsBuffer.destroy();
    stagingMassesBuffer.destroy();
    stagingfixedpointBuffer.destroy();
  }

  void prepareDescriptorPool() {
    // This is shared between graphics and compute
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 3),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              maxConcurrentFrames * 8),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            maxConcurrentFrames * 2)};
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
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        vks::initializers::descriptorSetLayoutBinding(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT, 1)};
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
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_TRUE);
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
        loadShader(getShadersPath() + "riddfmb2d/beam2d.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "riddfmb2d/beam2d.frag.spv",
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
                                              &graphics.pipelines.beam2d));
  }

  void Boundarycondition() {
    auto &verts = modelBeam2d.cpuVertices;
    compute.fixedpoint.resize(verts.size());
    for (int i = 0; i < verts.size(); i++) {
      if (verts[i].pos.x >= -0.01 && verts[i].pos.x <= 0.01) {
        compute.fixedpoint[i] = 1;
      } else {
        compute.fixedpoint[i] = 0;
      }
    }
  }
  void buildElementInfoFromMesh() {
    auto &verts = modelBeam2d.cpuVertices;
    auto &inds = modelBeam2d.cpuIndices;
    compute.elementInfo.clear();
    for (size_t i = 0; i + 2 < inds.size(); i += 3) {
      ElementInfo elem{};
      elem.elemId = static_cast<int>(compute.elementInfo.size());
      elem.pid = glm::ivec4(inds[i], inds[i + 1], inds[i + 2], 0);
      glm::vec2 p0(verts[inds[i]].pos.x, verts[inds[i]].pos.y);
      glm::vec2 p1(verts[inds[i + 1]].pos.x, verts[inds[i + 1]].pos.y);
      glm::vec2 p2(verts[inds[i + 2]].pos.x, verts[inds[i + 2]].pos.y);
      glm::vec2 e1 = p1 - p0;
      glm::vec2 e2 = p2 - p0;
      elem.restShape = glm::mat2(e1, e2);
      elem.restShapeInv = glm::inverse(elem.restShape);
      elem.restVol = 0.5f * std::abs(glm::determinant(elem.restShape));
      compute.elementInfo.push_back(elem);
    }
  }
  void precompute() {
    buildElementInfoFromMesh();
    uint32_t nelements = compute.elementInfo.size();
    auto &verts = modelBeam2d.cpuVertices;
    uint32_t nparticles = verts.size();
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
          for (int i = 0; i < 3; ++i) {
            if (particleOccupied[elemInfo.pid[i]]) {
              canAdd = false;
              break;
            }
          }
          if (canAdd) {
            // add to current set
            currentSet.emplace_back(*it);
            for (int i = 0; i <= 2; ++i) {
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
        for (int j = 0; j <= 2; ++j) {
          compute.masses[info.pid[j]] += mass / static_cast<float>(3.0);
        }
      }
    }
  }
  void prepareComputeParallel() {
    precompute();
    Boundarycondition();
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
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
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
            &storageBuffers.input.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            &storageBuffers.output.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
            &compute.uniformBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3,
            &compute.lambdaBuffer.descriptor),
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
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8,
            &compute.fixedpointBuffer.descriptor)};

    vkUpdateDescriptorSets(
        device, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
        computeWriteDescriptorSets.data(), 0, NULL);

    // Create pipelines (begin / solve / end)
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb2d/rid_begin.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.begin));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb2d/rid_solve.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.solve));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "riddfmb2d/rid_end.comp.spv",
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
      compute.uniformData.deltaT = fmin(frameTimer, 0.008) * 0.8;
    } else {
      compute.uniformData.deltaT = 0.0f;
    }
    memcpy(compute.uniformBuffer.mapped, &compute.uniformData,
           sizeof(Compute::UniformData));
  }

  void updateGraphicsUBO() {
    graphics.uniformData.projection = camera.matrices.perspective;
    graphics.uniformData.view = camera.matrices.view;
    memcpy(graphics.uniformBuffers[currentBuffer].mapped, &graphics.uniformData,
           sizeof(Graphics::UniformData));
  }
  void prepare() {
    VulkanExampleBase::prepare();
    // Check whether the compute queue family is distinct from the graphics
    // queue family
    dedicatedComputeQueue = vulkanDevice->queueFamilyIndices.graphics !=
                            vulkanDevice->queueFamilyIndices.compute;
    loadAssets();
    prepareComputeParallel();
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
    addComputeToGraphicsBarriers(
        cmdBuffer, 0, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

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
                      graphics.pipelines.beam2d);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0,
        1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
    vkCmdBindIndexBuffer(cmdBuffer, graphics.indices.buffer, 0,
                         VK_INDEX_TYPE_UINT32);
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &storageBuffers.output.buffer,
                           offsets);
    vkCmdDrawIndexed(cmdBuffer, indexCount, 1, 0, 0, 0);

    drawUI(cmdBuffer);

    vkCmdEndRenderPass(cmdBuffer);

    // release the storage buffers to the compute queue
    addGraphicsToComputeBarriers(cmdBuffer,
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void buildComputeCommandBuffer() {
    VkCommandBuffer cmdBuffer = compute.commandBuffers[currentBuffer];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Acquire the storage buffers from the graphics queue
    addGraphicsToComputeBarriers(cmdBuffer, 0, VK_ACCESS_SHADER_READ_BIT,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Single descriptor set, fixed binding of input/output buffers
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[0], 0, 0);

    // Stage 0: Begin solve
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipelines.begin);

    // Dispatch for all particles
    auto &verts = modelBeam2d.cpuVertices;
    uint32_t numParticles = verts.size();
    uint32_t numLambda = compute.lambdaData.size();
    uint32_t workgroupSizeX = 64;
    uint32_t numWorkgroupsX =
        (std::max(numParticles, numLambda) + workgroupSizeX - 1) /
        workgroupSizeX;
    vkCmdDispatch(cmdBuffer, numWorkgroupsX, 1, 1);

    // Barrier after begin solve
    addComputeToComputeBarriers(cmdBuffer);

    // Stage 1: Constraint solving
    // Iterate over all parallel sets and solve constraints
    const uint32_t numParallelSets =
        static_cast<uint32_t>(compute.elemParallelSlots.size()) - 1;
    const uint32_t constraintIterations = 10;

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipelines.solve);

    PushConstants pushConsts{};
    for (uint32_t iter = 0; iter < constraintIterations; iter++) {
      // Iterate over all parallel sets
      for (uint32_t setIdx = 0; setIdx < numParallelSets; setIdx++) {
        pushConsts.parallelSetStartIndex = setIdx;
        vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstants), &pushConsts);

        // Dispatch for this parallel set
        uint32_t setStart = compute.elemParallelSlots[setIdx];
        uint32_t setEnd = compute.elemParallelSlots[setIdx + 1];
        uint32_t setSize = setEnd - setStart;

        uint32_t workgroupSizeXSet = 64;
        uint32_t numWorkgroupsXSet =
            (setSize + workgroupSizeXSet - 1) / workgroupSizeXSet;
        vkCmdDispatch(cmdBuffer, numWorkgroupsXSet, 1, 1);

        // Barrier between parallel sets within same iteration
        if (setIdx < numParallelSets - 1) {
          addComputeToComputeBarriers(cmdBuffer);
        }
      }

      // Barrier between constraint iterations
      if (iter < constraintIterations - 1) {
        addComputeToComputeBarriers(cmdBuffer);
      }
    }

    // Stage 2: End solve
    // Update velocities based on position changes and write to particleOut
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipelines.end);

    // Dispatch for all particles
    uint32_t numParticlesStage2 = verts.size();
    uint32_t workgroupSizeXStage2 = 64;
    uint32_t numWorkgroupsXStage2 =
        (numParticlesStage2 + workgroupSizeXStage2 - 1) / workgroupSizeXStage2;
    vkCmdDispatch(cmdBuffer, numWorkgroupsXStage2, 1, 1);

    // Release the storage buffers back to the graphics queue
    addComputeToGraphicsBarriers(cmdBuffer, VK_ACCESS_SHADER_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmdBuffer);
  }

  virtual void render() {
    if (!prepared)
      return;
    // Submit compute commands
    {
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));
      VK_CHECK_RESULT(vkResetFences(device, 1, &compute.fences[currentBuffer]));

      updateComputeUBO();
      buildComputeCommandBuffer();

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

      // Wait for compute to finish, then read back particle[525] position
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));
    }

    // Submit graphics commands
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
  }
  virtual void writeMesh() { int k; }
};

VULKAN_EXAMPLE_MAIN()
