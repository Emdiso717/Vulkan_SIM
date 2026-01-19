#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"
#include <algorithm>
#include <iostream>

class VulkanExample : public VulkanExampleBase {
public:
  uint32_t indexCount{0};
  uint32_t readSet{0};
  bool simulateWind{false};
  bool dedicatedComputeQueue{false};

  vks::Texture2D textureCloth;
  vkglTF::Model modelSphere;

  struct Particle {
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 uv;
    glm::vec4 normal;
  };

  struct ElementInfo {
    int elemId;
    glm::vec<2, int> pid;
    float restLength{0.0f};
  };

  // Push constants structure for compute shader
  struct PushConstants {
    uint32_t computeStage; // 0=beginSolve, 1=constraintSolve, 2=endSolve
    uint32_t parallelSetStartIndex; // Starting index of current parallel set in
                                    // elementInfo
  };

  struct Cloth {
    glm::uvec2 gridsize{60, 60};
    glm::vec2 size{5.0f, 5.0f};
  } cloth;

  struct StorageBuffers {
    vks::Buffer input;
    vks::Buffer output;
    vks::Buffer lambda; // Lambda buffer for XPBD constraint solving
  } storageBuffers;

  struct Graphics {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline cloth{VK_NULL_HANDLE};
      VkPipeline sphere{VK_NULL_HANDLE};
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
    VkPipeline pipeline{VK_NULL_HANDLE};
    struct UniformData {
      float deltaT{0.0f};
      float particleMass{0.1f};
      float springStiffness{2000.0f};
      float damping{0.25f};
      float sphereRadius{1.0f};
      glm::vec4 spherePos{0.0f, 0.0f, 0.0f, 0.0f};
      glm::vec4 gravity{0.0f, 9.8f, 0.0f, 0.0f};
      glm::ivec2 particleCount{0};
    } uniformData;
    vks::Buffer uniformBuffer;
    // ElementInfo and elemParallelSlots need to be SSBOs, not UBO
    // because UBO doesn't support dynamic arrays
    std::vector<ElementInfo> elementInfo{};
    std::vector<int> elemParallelSlots{};
    vks::Buffer elementInfoBuffer;
    vks::Buffer elemParallelSlotsBuffer;
  } compute;

  VulkanExample() : VulkanExampleBase() {
    title = "XPBD Cloth Simulation";
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
      vkDestroyPipeline(device, graphics.pipelines.cloth, nullptr);
      vkDestroyPipeline(device, graphics.pipelines.sphere, nullptr);
      vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout,
                                   nullptr);
      textureCloth.destroy();

      // Compute
      compute.uniformBuffer.destroy();
      compute.elementInfoBuffer.destroy();
      compute.elemParallelSlotsBuffer.destroy();
      vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout,
                                   nullptr);
      vkDestroyPipeline(device, compute.pipeline, nullptr);
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
      storageBuffers.lambda.destroy();
    }
  }

  // Enable physical device features required for this example
  virtual void getEnabledFeatures() {
    if (deviceFeatures.samplerAnisotropy) {
      enabledFeatures.samplerAnisotropy = VK_TRUE;
    }
  };
  // Load the assets for the example
  void loadAssets() {
    const uint32_t glTFLoadingFlags =
        vkglTF::FileLoadingFlags::PreTransformVertices |
        vkglTF::FileLoadingFlags::PreMultiplyVertexColors |
        vkglTF::FileLoadingFlags::FlipY;
    modelSphere.loadFromFile(getAssetPath() + "models/sphere.gltf",
                             vulkanDevice, queue, glTFLoadingFlags);
    textureCloth.loadFromFile(getAssetPath() + "textures/vulkan_cloth_rgba.ktx",
                              VK_FORMAT_R8G8B8A8_UNORM, vulkanDevice, queue);
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
  void addComputeToComputeBarriers(VkCommandBuffer commandBuffer,
                                   uint32_t readSet) {
    VkBufferMemoryBarrier bufferBarrier =
        vks::initializers::bufferMemoryBarrier();
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.size = VK_WHOLE_SIZE;
    std::vector<VkBufferMemoryBarrier> bufferBarriers;
    if (readSet == 0) {
      // SRS - we have written to output.buffer and need a memory barrier before
      // reading it
      //	   - don't need a memory barrier for input.buffer, the execution
      // barrier is enough
      bufferBarrier.buffer = storageBuffers.output.buffer;
      bufferBarriers.push_back(bufferBarrier);
    } else // if (readSet == 1)
    {
      // SRS - we have written to input.buffer and need a memory barrier before
      // reading it
      //	   - don't need a memory barrier for output.buffer, the
      // execution barrier is enough
      bufferBarrier.buffer = storageBuffers.input.buffer;
      bufferBarriers.push_back(bufferBarrier);
    }
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
  void prepareStorageBuffers() {
    std::vector<Particle> particleBuffer(cloth.gridsize.x * cloth.gridsize.y);

    float dx = cloth.size.x / (cloth.gridsize.x - 1);
    float dy = cloth.size.y / (cloth.gridsize.y - 1);
    float du = 1.0f / (cloth.gridsize.x - 1);
    float dv = 1.0f / (cloth.gridsize.y - 1);

    // Set up a flat cloth that falls onto sphere
    glm::mat4 transM =
        glm::translate(glm::mat4(1.0f), glm::vec3(-cloth.size.x / 2.0f, -2.0f,
                                                  -cloth.size.y / 2.0f));
    for (uint32_t i = 0; i < cloth.gridsize.y; i++) {
      for (uint32_t j = 0; j < cloth.gridsize.x; j++) {
        particleBuffer[i + j * cloth.gridsize.y].pos =
            transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
        particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f);
        particleBuffer[i + j * cloth.gridsize.y].uv =
            glm::vec4(1.0f - du * i, dv * j, 0.0f, 0.0f);
      }
    }

    VkDeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);

    // Staging
    // SSBO won't be changed on the host after upload so copy to device local
    // memory

    vks::Buffer stagingBuffer;

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, storageBufferSize,
                               particleBuffer.data());

    // SSBOs will be used both as storage buffers (compute) and vertex buffers
    // (graphics)
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

    // Copy from staging buffer
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = storageBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.output.buffer,
                    1, &copyRegion);
    // Add an initial release barrier to the graphics queue,
    // so that when the compute command buffer executes for the first time
    // it doesn't complain about a lack of a corresponding "release" to its
    // "acquire"
    addGraphicsToComputeBarriers(copyCmd, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 0,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();

    // Indices
    std::vector<uint32_t> indices;
    for (uint32_t y = 0; y < cloth.gridsize.y - 1; y++) {
      for (uint32_t x = 0; x < cloth.gridsize.x; x++) {
        indices.push_back((y + 1) * cloth.gridsize.x + x);
        indices.push_back((y)*cloth.gridsize.x + x);
      }
      // Primitive restart (signaled by special value 0xFFFFFFFF)
      indices.push_back(0xFFFFFFFF);
    }
    uint32_t indexBufferSize =
        static_cast<uint32_t>(indices.size()) * sizeof(uint32_t);
    indexCount = static_cast<uint32_t>(indices.size());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingBuffer, indexBufferSize, indices.data());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &graphics.indices, indexBufferSize);

    // Copy from staging buffer
    copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                true);
    copyRegion = {};
    copyRegion.size = indexBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, graphics.indices.buffer, 1,
                    &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    stagingBuffer.destroy();
  }

  void prepareDescriptorPool() {
    // This is shared between graphics and compute
    // Compute uses: 2 SSBOs (particleIn/Out) + 1 UBO + 1 lambda SSBO = 4 SSBOs
    // + 1 UBO per frame Graphics uses: 1 UBO + 1 image sampler = 1 UBO + 1
    // image per frame Total: 2 UBOs + 4 SSBOs + 1 image per frame
    // (maxConcurrentFrames)
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 3),
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            maxConcurrentFrames *
                7), // 2 particle + 1 lambda + 2 elementInfo + 2
                    // elemParallelSlots per compute set (2 sets) + 1 buffer
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            maxConcurrentFrames * 2)};
    VkDescriptorPoolCreateInfo descriptorPoolInfo =
        vks::initializers::descriptorPoolCreateInfo(poolSizes,
                                                    maxConcurrentFrames * 3);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr,
                                           &descriptorPool));
  }
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
              &graphics.uniformBuffers[i].descriptor),
          vks::initializers::writeDescriptorSet(
              graphics.descriptorSets[i],
              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
              &textureCloth.descriptor)};
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
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, 0, VK_TRUE);
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
        loadShader(getShadersPath() + "xpbdcloth/cloth.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "xpbdcloth/cloth.frag.spv",
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
                                              &graphics.pipelines.cloth));

    // Sphere rendering pipeline
    pipelineCreateInfo.pVertexInputState =
        vkglTF::Vertex::getPipelineVertexInputState(
            {vkglTF::VertexComponent::Position, vkglTF::VertexComponent::UV,
             vkglTF::VertexComponent::Normal});
    inputState.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(inputAttributes.size());
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyState.primitiveRestartEnable = VK_FALSE;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    shaderStages = {loadShader(getShadersPath() + "xpbdcloth/sphere.vert.spv",
                               VK_SHADER_STAGE_VERTEX_BIT),
                    loadShader(getShadersPath() + "xpbdcloth/sphere.frag.spv",
                               VK_SHADER_STAGE_FRAGMENT_BIT)};
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics.pipelines.sphere));
  }
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

    // Set some initial values
    float dx = cloth.size.x / (cloth.gridsize.x - 1);
    float dy = cloth.size.y / (cloth.gridsize.y - 1);
    compute.uniformData.particleCount = cloth.gridsize;
    // Initialize ElementInfo for cloth constraints (spring connections)
    // Each particle connects to its 8 neighbors (horizontal, vertical,
    // diagonal)
    float restDistH = dx;                       // Horizontal rest distance
    float restDistV = dy;                       // Vertical rest distance
    float restDistD = sqrtf(dx * dx + dy * dy); // Diagonal rest distance

    compute.elementInfo.clear();
    int elemId = 0;

    // For each particle, create connections to all 8 neighbors
    // Note: Index formula must match prepareStorageBuffers: index = y + x *
    // gridsize.y
    for (uint32_t y = 0; y < cloth.gridsize.y; y++) {
      for (uint32_t x = 0; x < cloth.gridsize.x; x++) {
        int currentIdx = y + x * cloth.gridsize.y;

        // Right neighbor (horizontal)
        if (x < cloth.gridsize.x - 1) {
          int rightIdx = y + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, rightIdx);
          elem.restLength = restDistH;
          compute.elementInfo.push_back(elem);
        }

        // Left neighbor (horizontal) - only create if not already created by
        // right neighbor
        if (x > 0) {
          int leftIdx = y + (x - 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, leftIdx);
          elem.restLength = restDistH;
          compute.elementInfo.push_back(elem);
        }

        // Bottom neighbor (vertical)
        if (y < cloth.gridsize.y - 1) {
          int bottomIdx = (y + 1) + x * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomIdx);
          elem.restLength = restDistV;
          compute.elementInfo.push_back(elem);
        }

        // Top neighbor (vertical) - only create if not already created by
        // bottom neighbor
        if (y > 0) {
          int topIdx = (y - 1) + x * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, topIdx);
          elem.restLength = restDistV;
          compute.elementInfo.push_back(elem);
        }

        // Bottom-right neighbor (diagonal)
        if (x < cloth.gridsize.x - 1 && y < cloth.gridsize.y - 1) {
          int bottomRightIdx = (y + 1) + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomRightIdx);
          elem.restLength = restDistD;
          compute.elementInfo.push_back(elem);
        }

        // Bottom-left neighbor (diagonal)
        if (x > 0 && y < cloth.gridsize.y - 1) {
          int bottomLeftIdx = (y + 1) + (x - 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomLeftIdx);
          elem.restLength = restDistD;
          compute.elementInfo.push_back(elem);
        }

        // Top-right neighbor (diagonal)
        if (x < cloth.gridsize.x - 1 && y > 0) {
          int topRightIdx = (y - 1) + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, topRightIdx);
          elem.restLength = restDistD;
          compute.elementInfo.push_back(elem);
        }

        // Top-left neighbor (diagonal)
        if (x > 0 && y > 0) {
          int topLeftIdx = (y - 1) + (x - 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, topLeftIdx);
          elem.restLength = restDistD;
          compute.elementInfo.push_back(elem);
        }
      }
    }

    // Precompute parallelable element sets for efficient GPU processing
    // Elements that don't share particles can be processed in parallel
    {
      std::cout
          << "Precompute Stage: Precomputing Elements Parallelable Sets..."
          << std::endl;
      auto &elemInfos = compute.elementInfo;
      auto nElements = [&]() { return elemInfos.size(); };
      auto nParticles = [&]() { return cloth.gridsize.x * cloth.gridsize.y; };

      std::vector<int> elementIds;
      elementIds.reserve(elemInfos.size());
      for (size_t i = 0; i < nElements(); ++i) {
        elementIds.emplace_back(static_cast<int>(i));
      }

      std::vector<std::vector<int>> elemParaSets;
      while (!elementIds.empty()) {
        std::vector<bool> particleOccupied(nParticles(), false);
        std::vector<int> currentSet;
        for (auto it = elementIds.begin(); it != elementIds.end();) {
          const auto &elemInfo = elemInfos[*it];
          // Check if both particles are not occupied
          bool canAdd = true;
          for (int i = 0; i < 2; ++i) {
            if (particleOccupied[elemInfo.pid[i]]) {
              canAdd = false;
              break;
            }
          }
          if (canAdd) {
            // add to current set
            currentSet.emplace_back(*it);
            for (int i = 0; i < 2; ++i) {
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
      std::cout << "\tElements Parallelable Sets Precomputed." << std::endl;
      std::cout << "\tTotal " << elemParaSets.size() << " sets for "
                << nElements() << " elements." << std::endl;

      // reorder elemInfos according to parallelable sets
      std::vector<ElementInfo> reorderedElemInfos;
      reorderedElemInfos.reserve(nElements());
      compute.elemParallelSlots.clear();
      for (const auto &elemIdSet : elemParaSets) {
        compute.elemParallelSlots.emplace_back(
            static_cast<int>(reorderedElemInfos.size()));
        for (const auto elemId : elemIdSet) {
          reorderedElemInfos.emplace_back(elemInfos[elemId]);
          // correct elemId
          reorderedElemInfos.back().elemId =
              static_cast<int>(reorderedElemInfos.size()) - 1;
        }
      }
      compute.elemParallelSlots.emplace_back(
          static_cast<int>(reorderedElemInfos.size()));
      assert(reorderedElemInfos.size() == nElements());
      std::swap(elemInfos, reorderedElemInfos);
    }

    // Create elementInfo and elemParallelSlots buffers (SSBOs)
    uint32_t numElements = static_cast<uint32_t>(compute.elementInfo.size());
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

    // Copy elementInfo and elemParallelSlots from staging to device
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};
    copyRegion.size = elementInfoBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElementInfoBuffer.buffer,
                    compute.elementInfoBuffer.buffer, 1, &copyRegion);
    copyRegion.size = elemParallelSlotsBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElemParallelSlotsBuffer.buffer,
                    compute.elemParallelSlotsBuffer.buffer, 1, &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
    stagingElementInfoBuffer.destroy();
    stagingElemParallelSlotsBuffer.destroy();

    // Create lambda buffer for XPBD constraint solving
    // One float per constraint (element)
    VkDeviceSize lambdaBufferSize = numElements * sizeof(float);

    // Initialize lambda buffer with zeros using staging buffer
    std::vector<float> lambdaData(numElements, 0.0f);
    vks::Buffer stagingLambdaBuffer;
    vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               &stagingLambdaBuffer, lambdaBufferSize,
                               lambdaData.data());

    vulkanDevice->createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &storageBuffers.lambda, lambdaBufferSize);

    // Copy lambda buffer from staging to device
    copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                true);
    copyRegion = {};
    copyRegion.size = lambdaBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingLambdaBuffer.buffer,
                    storageBuffers.lambda.buffer, 1, &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
    stagingLambdaBuffer.destroy();

    // Create compute pipeline
    // Binding layout:
    // 0: ParticleIn (SSBO)
    // 1: ParticleOut (SSBO)
    // 2: UBO (Uniform Buffer)
    // 3: Lambda (SSBO)
    // 4: ElementInfo (SSBO)
    // 5: ElemParallelSlots (SSBO)
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
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute.descriptorSetLayout, 1);

    // Push constants used to pass some parameters
    // We need to pass: computeStage (uint32_t) and parallelSetStartIndex
    // (uint32_t)
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

    // Create two descriptor sets with input and output buffers switched
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                             &compute.descriptorSets[0]));
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo,
                                             &compute.descriptorSets[1]));

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
            &storageBuffers.lambda.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
            &compute.elementInfoBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5,
            &compute.elemParallelSlotsBuffer.descriptor),

        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0,
            &storageBuffers.output.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            &storageBuffers.input.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2,
            &compute.uniformBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3,
            &storageBuffers.lambda.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4,
            &compute.elementInfoBuffer.descriptor),
        vks::initializers::writeDescriptorSet(
            compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5,
            &compute.elemParallelSlotsBuffer.descriptor)};

    vkUpdateDescriptorSets(
        device, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
        computeWriteDescriptorSets.data(), 0, NULL);

    // Create pipeline
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "xpbdcloth/cloth.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1,
                                             &computePipelineCreateInfo,
                                             nullptr, &compute.pipeline));

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

  void prepare() {
    VulkanExampleBase::prepare();
    dedicatedComputeQueue = vulkanDevice->queueFamilyIndices.graphics !=
                            vulkanDevice->queueFamilyIndices.compute;
    loadAssets();
    prepareStorageBuffers();
    prepareDescriptorPool();
    prepareGraphics();
    prepareCompute();
    prepared = true;
  }

  void updateComputeUBO() {
    if (!paused) {
      // SRS - Clamp frameTimer to max 20ms refresh period (e.g. if blocked on
      // resize), otherwise image breakup can occur
      compute.uniformData.deltaT = fmin(frameTimer, 0.02f) * 0.0025f;

      if (simulateWind) {
        std::default_random_engine rndEngine(
            benchmark.active ? 0 : (unsigned)time(nullptr));
        std::uniform_real_distribution<float> rd(1.0f, 12.0f);
        compute.uniformData.gravity.x = cos(glm::radians(-timer * 360.0f)) *
                                        (rd(rndEngine) - rd(rndEngine));
        compute.uniformData.gravity.z =
            sin(glm::radians(timer * 360.0f)) * (rd(rndEngine) - rd(rndEngine));
      } else {
        compute.uniformData.gravity.x = 0.0f;
        compute.uniformData.gravity.z = 0.0f;
      }
    } else {
      compute.uniformData.deltaT = 0.0f;
    }
    // Copy only the fixed-size UniformData structure (without vectors)
    memcpy(compute.uniformBuffer.mapped, &compute.uniformData,
           sizeof(Compute::UniformData));
  }

  void updateGraphicsUBO() {
    graphics.uniformData.projection = camera.matrices.perspective;
    graphics.uniformData.view = camera.matrices.view;
    memcpy(graphics.uniformBuffers[currentBuffer].mapped, &graphics.uniformData,
           sizeof(Graphics::UniformData));
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

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipeline);

    // ========== BEGIN SOLVE: Predict positions ==========
    // Stage 0: Reset lambda, predict positions for free particles
    // For constrained particles: vel = 0
    // For free particles:
    //   - vel stores old position temporarily
    //   - pos = x + v*dt + dt²*g (predicted position)
    PushConstants pushConsts;
    pushConsts.computeStage = 0;          // 0 = beginSolve
    pushConsts.parallelSetStartIndex = 0; // Not used in beginSolve
    vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                       &pushConsts);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[readSet], 0, 0);
    // Dispatch for all particles (parallel)
    vkCmdDispatch(cmdBuffer, cloth.gridsize.x / 10, cloth.gridsize.y / 10, 1);
    addComputeToComputeBarriers(cmdBuffer, readSet);

    // ========== CONSTRAINT SOLVE: XPBD iterations ==========
    // Stage 1: Constraint solving iterations (using parallel sets)
    // Process constraints in parallel sets - each set contains constraints
    // that don't share particles, so they can be processed in parallel
    pushConsts.computeStage = 1; // 1 = constraintSolve
    const uint32_t iterations = 64;
    for (uint32_t j = 0; j < iterations; j++) {
      readSet = 1 - readSet; // Ping-pong buffer
      const auto nParallelSets =
          static_cast<uint32_t>(compute.elemParallelSlots.size() - 1);

      // Dispatch each parallel set independently
      // Elements in the same set don't share particles, so they can run in
      // parallel
      for (uint32_t setIdx = 0; setIdx < nParallelSets; setIdx++) {
        uint32_t setStart = compute.elemParallelSlots[setIdx];
        uint32_t setEnd = compute.elemParallelSlots[setIdx + 1];
        uint32_t numElementsInSet = setEnd - setStart;

        if (numElementsInSet > 0) {
          // Pass the starting index of current parallel set to shader
          pushConsts.parallelSetStartIndex = setStart;
          vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(PushConstants), &pushConsts);
          vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  compute.pipelineLayout, 0, 1,
                                  &compute.descriptorSets[readSet], 0, 0);

          // Dispatch only the elements in this parallel set
          uint32_t workgroupSize = 64; // Match shader local_size_x
          uint32_t dispatchX =
              (numElementsInSet + workgroupSize - 1) / workgroupSize;
          vkCmdDispatch(cmdBuffer, dispatchX, 1, 1);
        }
      }

      // Barrier between iterations (except last)
      if (j != iterations - 1) {
        addComputeToComputeBarriers(cmdBuffer, readSet);
      }
    }

    // ========== END SOLVE: Update velocities ==========
    // Stage 2: Update velocities from position changes
    // v = (x_new - x_old) / dt, where x_old was stored in vel
    // For constrained particles: vel = 0
    pushConsts.computeStage = 2;          // 2 = endSolve
    pushConsts.parallelSetStartIndex = 0; // Not used in endSolve
    vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                       &pushConsts);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[readSet], 0, 0);
    // Dispatch for all particles (parallel)
    vkCmdDispatch(cmdBuffer, cloth.gridsize.x / 10, cloth.gridsize.y / 10, 1);

    // Release the storage buffers back to the graphics queue
    addComputeToGraphicsBarriers(cmdBuffer, VK_ACCESS_SHADER_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    vkEndCommandBuffer(cmdBuffer);
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

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport =
        vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    VkDeviceSize offsets[1] = {0};

    // Render sphere
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics.pipelines.sphere);
    vkCmdBindDescriptorSets(
        cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0,
        1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
    modelSphere.draw(cmdBuffer);

    // Render cloth
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      graphics.pipelines.cloth);
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

    // Release the storage buffers to the compute queue
    addGraphicsToComputeBarriers(cmdBuffer,
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
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

  virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay) {
    if (overlay->header("Settings")) {
      overlay->checkBox("Simulate wind", &simulateWind);
    }
  }
};

VULKAN_EXAMPLE_MAIN()
