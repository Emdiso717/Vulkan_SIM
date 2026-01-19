#ifdef _WIN32
#pragma comment(linker, "/subsystem:console")
#endif

#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"

class VulkanExample : public VulkanExampleBase {
public:
  uint32_t readSet{0};
  uint32_t indexCount{0};
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
    alignas(8) glm::vec<2, int> pid;
    float restLength{0.0f};
  };

  struct Cloth {
    glm::uvec2 gridsize{60, 60};
    glm::vec2 size{5.0f, 5.0f};
  } cloth;

  struct StorageBuffers {
    vks::Buffer input;
    vks::Buffer output;
  } storageBuffers;

  struct PushConstants {
    uint32_t computeStage;
    uint32_t parallelSetStartIndex;
  };

  // Resources for the graphics part of the example
  struct Graphics {
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    struct Pipelines {
      VkPipeline cloth{VK_NULL_HANDLE};
      VkPipeline sphere{VK_NULL_HANDLE};
    } pipelines;
    // The vertices will be stored in the shader storage buffers, so we only
    // need an index buffer in this structure
    vks::Buffer indices;
    struct UniformData {
      glm::mat4 projection;
      glm::mat4 view;
      glm::vec4 lightPos{-2.0f, 4.0f, -2.0f, 1.0f};
    } uniformData;
    std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;
  } graphics;

  // Resources for the compute part of the example
  // Number of compute command buffers: set to 1 for serialized processing or 2
  // for in-parallel with graphics queue
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
      // These arguments define the spring setup for the cloth piece
      // Changing these changes how the cloth reacts
      float particleMass{0.1f};
      float springStiffness{2000.0f};
      float damping{0.25f};
      float restDistH{0};
      float restDistV{0};
      float restDistD{0};
      float sphereRadius{1.0f};
      glm::vec4 spherePos{0.0f, 0.0f, 0.0f, 0.0f};
      glm::vec4 gravity{0.0f, 9.8f, 0.0f, 0.0f};
      glm::ivec2 particleCount{0};
    } uniformData;
    std::vector<ElementInfo> elementInfo;
    std::vector<float> lambdaData;
    std::vector<int> elemParallelSlots;
    // No need to duplicate Only set up once at application start
    vks::Buffer uniformBuffer;
    vks::Buffer lambdaBuffer;
    vks::Buffer elementInfoBuffer;
    vks::Buffer elemParallelSlotsBuffer;
  } compute;

  VulkanExample() : VulkanExampleBase() {
    title = "Compute shader cloth simulation";
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
      compute.lambdaBuffer.destroy();
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

  // Setup and fill the shader storage buffers containing the particles
  // These buffers are used as shader storage buffers in the compute shader (to
  // update them) and as vertex input in the vertex shader (to display them)
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
    // Copy initial data to both input and output buffers
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.input.buffer,
                    1, &copyRegion);
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
    // Compute descriptor sets need:
    //   - 2 descriptor sets (ping-pong) * maxConcurrentFrames
    //   - Each set has: 2 STORAGE_BUFFER (input/output) + 1 UNIFORM_BUFFER + 3
    //   STORAGE_BUFFER (lambda, elementInfo, elemParallelSlots)
    //   - Total: maxConcurrentFrames * 2 * (2 + 3) = maxConcurrentFrames * 10
    //   STORAGE_BUFFER
    //   - Total: maxConcurrentFrames * 2 * 1 = maxConcurrentFrames * 2
    //   UNIFORM_BUFFER
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames *
                                                  3), // Graphics + Compute
        vks::initializers::descriptorPoolSize(
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            maxConcurrentFrames * 10 +
                maxConcurrentFrames *
                    4), // Compute (10) + Graphics (4 if needed)
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
        loadShader(getShadersPath() + "computecloth/cloth.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "computecloth/cloth.frag.spv",
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
    shaderStages = {
        loadShader(getShadersPath() + "computecloth/sphere.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "computecloth/sphere.frag.spv",
                   VK_SHADER_STAGE_FRAGMENT_BIT)};
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics.pipelines.sphere));
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

    // Set some initial values
    float dx = cloth.size.x / (cloth.gridsize.x - 1);
    float dy = cloth.size.y / (cloth.gridsize.y - 1);

    compute.uniformData.restDistH = dx;
    compute.uniformData.restDistV = dy;
    compute.uniformData.restDistD = sqrtf(dx * dx + dy * dy);
    compute.uniformData.particleCount = cloth.gridsize;

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
          elem.restLength = compute.uniformData.restDistH;
          compute.elementInfo.push_back(elem);
        }

        // Bottom neighbor (vertical)
        if (y < cloth.gridsize.y - 1) {
          int bottomIdx = (y + 1) + x * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomIdx);
          elem.restLength = compute.uniformData.restDistV;
          compute.elementInfo.push_back(elem);
        }

        // Bottom-right neighbor (diagonal)
        if (x < cloth.gridsize.x - 1 && y < cloth.gridsize.y - 1) {
          int bottomRightIdx = (y + 1) + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomRightIdx);
          elem.restLength = compute.uniformData.restDistD;
          compute.elementInfo.push_back(elem);
        }

        // Top-left neighbor (diagonal)
        if (x > 0 && y > 0) {
          int topLeftIdx = (y - 1) + (x - 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, topLeftIdx);
          elem.restLength = compute.uniformData.restDistD;
          compute.elementInfo.push_back(elem);
        }
      }
    }

    // Precompute parallel sets before creating buffers
    std::cout << "Precompute Stage: Precomputing Elements Parallelable Sets..."
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
    // COUT Parallel Sets Verification
    {
      // Debug output: Verify correctness of parallel sets
      std::cout << "\n=== Debug: Parallel Sets Verification ===" << std::endl;

      // 1. Check each set size and show first few sets
      uint32_t totalElementsInSets = 0;
      uint32_t maxSetSize = 0;
      uint32_t minSetSize = UINT32_MAX;
      for (size_t i = 0; i < elemParaSets.size(); ++i) {
        uint32_t setSize = static_cast<uint32_t>(elemParaSets[i].size());
        totalElementsInSets += setSize;
        if (setSize > maxSetSize)
          maxSetSize = setSize;
        if (setSize < minSetSize)
          minSetSize = setSize;

        // Show first 5 sets and last set for verification
        if (i < 5 || i == elemParaSets.size() - 1) {
          std::cout << "\tSet[" << i << "] size: " << setSize;
          if (setSize <= 10) {
            std::cout << " [elements: ";
            for (auto elemId : elemParaSets[i]) {
              std::cout << elemId << " ";
            }
            std::cout << "]";
          }
          std::cout << std::endl;
        }
      }
      std::cout << "\tSet size range: " << minSetSize << " - " << maxSetSize
                << std::endl;
      std::cout << "\tTotal elements in all sets: " << totalElementsInSets
                << " (expected: " << nElements() << ")" << std::endl;

      // 2. Verify no particle conflict within each set
      bool hasConflict = false;
      for (size_t setIdx = 0; setIdx < elemParaSets.size(); ++setIdx) {
        std::vector<bool> particleOccupied(nParticles(), false);
        for (auto elemId : elemParaSets[setIdx]) {
          const auto &elemInfo = elemInfos[elemId];
          for (int i = 0; i < 2; ++i) {
            int pid = elemInfo.pid[i];
            if (particleOccupied[pid]) {
              std::cout << "\tERROR: Set[" << setIdx
                        << "] has conflict! Particle " << pid
                        << " appears in multiple constraints." << std::endl;
              hasConflict = true;
            }
            particleOccupied[pid] = true;
          }
        }
      }
      if (!hasConflict) {
        std::cout << "\t✓ No particle conflicts within sets (correct!)"
                  << std::endl;
      }

      // 3. Verify all elements are present (no duplicates, no missing)
      std::vector<bool> elementUsed(nElements(), false);
      for (const auto &set : elemParaSets) {
        for (auto elemId : set) {
          if (elemId < 0 || elemId >= static_cast<int>(nElements())) {
            std::cout << "\tERROR: Invalid element ID " << elemId << " in sets!"
                      << std::endl;
          } else if (elementUsed[elemId]) {
            std::cout << "\tERROR: Element " << elemId
                      << " appears in multiple sets!" << std::endl;
          } else {
            elementUsed[elemId] = true;
          }
        }
      }
      bool allElementsPresent = true;
      for (size_t i = 0; i < nElements(); ++i) {
        if (!elementUsed[i]) {
          std::cout << "\tERROR: Element " << i << " is missing from all sets!"
                    << std::endl;
          allElementsPresent = false;
        }
      }
      if (allElementsPresent && totalElementsInSets == nElements()) {
        std::cout << "\t✓ All elements present exactly once (correct!)"
                  << std::endl;
      }

      // 4. Show sample constraints info
      std::cout << "\n=== Sample Constraints Info ===" << std::endl;
      for (size_t i = 0; i < std::min(size_t(20), elemInfos.size()); ++i) {
        std::cout << "\tConstraint[" << i << "]: pid=(" << elemInfos[i].pid[0]
                  << "," << elemInfos[i].pid[1]
                  << "), restLength=" << elemInfos[i].restLength << std::endl;
      }

      std::cout << "=== End Debug Output ===\n" << std::endl;
      std::cout.flush(); // Force immediate output
    }

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
    // COUT Sample Constraints Info
    {
      std::cout << "\n=== Sample Constraints Info ===" << std::endl;
      for (size_t i = 0; i < std::min(size_t(20), compute.elementInfo.size());
           ++i) {
        std::cout << "\tConstraint[" << i << "]: pid=("
                  << compute.elementInfo[i].pid[0] << ","
                  << compute.elementInfo[i].pid[1]
                  << "), restLength=" << compute.elementInfo[i].restLength
                  << std::endl;
      }

      // Debug output: Verify reordering
      std::cout << "\n=== Debug: Reordering Verification ===" << std::endl;
      std::cout << "\telemParallelSlots size: "
                << compute.elemParallelSlots.size()
                << " (expected: " << elemParaSets.size() + 1 << ")"
                << std::endl;
      std::cout << "\tReordered elementInfo size: " << elemInfos.size()
                << " (expected: " << nElements() << ")" << std::endl;

      // Show parallel slots (first 10 and last few)
      std::cout << "\tParallel slots (first 10): ";
      for (size_t i = 0;
           i < std::min(size_t(10), compute.elemParallelSlots.size()); ++i) {
        std::cout << compute.elemParallelSlots[i] << " ";
      }
      std::cout << std::endl;
      if (compute.elemParallelSlots.size() > 10) {
        std::cout << "\tParallel slots (last 5): ";
        for (size_t i = compute.elemParallelSlots.size() - 5;
             i < compute.elemParallelSlots.size(); ++i) {
          std::cout << compute.elemParallelSlots[i] << " ";
        }
        std::cout << std::endl;
      }
      // Verify parallel slots are monotonic
      bool slotsValid = true;
      for (size_t i = 1; i < compute.elemParallelSlots.size(); ++i) {
        if (compute.elemParallelSlots[i] < compute.elemParallelSlots[i - 1]) {
          std::cout << "\tERROR: Parallel slots not monotonic! Slot[" << i - 1
                    << "]=" << compute.elemParallelSlots[i - 1] << " > Slot["
                    << i << "]=" << compute.elemParallelSlots[i] << std::endl;
          slotsValid = false;
        }
      }
      if (slotsValid) {
        std::cout << "\t✓ Parallel slots are monotonic (correct!)" << std::endl;
      }

      // Verify last slot equals total elements
      if (compute.elemParallelSlots.back() == static_cast<int>(nElements())) {
        std::cout << "\t✓ Last slot = " << compute.elemParallelSlots.back()
                  << " = total elements (correct!)" << std::endl;
      } else {
        std::cout << "\tERROR: Last slot = " << compute.elemParallelSlots.back()
                  << " != total elements = " << nElements() << std::endl;
      }

      std::cout << "=== End Reordering Verification ===\n" << std::endl;
      std::cout.flush();
    }

    // Initialize lambda data with zeros (one float per constraint)
    uint32_t numElements = static_cast<uint32_t>(compute.elementInfo.size());
    compute.lambdaData.resize(numElements, 0.0f);

    // Create all buffers at once (after data preparation is complete)
    // 1. ElementInfo buffer (SSBO)
    // Debug: Verify struct size matches GLSL std430 layout (should be 20 bytes)
    std::cout << "ElementInfo struct size: " << sizeof(ElementInfo)
              << " bytes (expected: 20 for std430 layout)" << std::endl;
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

    // 2. Lambda buffer (SSBO)
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

    // 3. ElemParallelSlots buffer (SSBO)
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

    // Copy all buffers from staging to device in one command buffer
    VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkBufferCopy copyRegion = {};

    // Copy elementInfo
    copyRegion.size = elementInfoBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElementInfoBuffer.buffer,
                    compute.elementInfoBuffer.buffer, 1, &copyRegion);

    // Copy lambda
    copyRegion.size = lambdaBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingLambdaBuffer.buffer,
                    compute.lambdaBuffer.buffer, 1, &copyRegion);

    // Copy elemParallelSlots
    copyRegion.size = elemParallelSlotsBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElemParallelSlotsBuffer.buffer,
                    compute.elemParallelSlotsBuffer.buffer, 1, &copyRegion);

    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

    // Clean up staging buffers
    stagingElementInfoBuffer.destroy();
    stagingLambdaBuffer.destroy();
    stagingElemParallelSlotsBuffer.destroy();

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
    };

    VkDescriptorSetLayoutCreateInfo descriptorLayout =
        vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(
        device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        vks::initializers::pipelineLayoutCreateInfo(
            &compute.descriptorSetLayout, 1);

    // Push constants used to pass some parameters
    // PushConstants has: computeStage (uint32_t) + parallelSetStartIndex
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
            &compute.lambdaBuffer.descriptor),
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
            &compute.lambdaBuffer.descriptor),
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
    computePipelineCreateInfo.stage = loadShader(
        getShadersPath() + "xpbd/cloth.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
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

  void updateComputeUBO() {
    if (!paused) {
      // SRS - Clamp frameTimer to max 20ms refresh period (e.g. if blocked on
      // resize), otherwise image breakup can occur
      compute.uniformData.deltaT = fmin(frameTimer, 0.02f) * 0.25f;

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
#if defined(_WIN32)
    // Setup console to see std::cout output
    setupConsole("XPBD Cloth Simulation");
#endif
    VulkanExampleBase::prepare();
    // Check whether the compute queue family is distinct from the graphics
    // queue family
    dedicatedComputeQueue = vulkanDevice->queueFamilyIndices.graphics !=
                            vulkanDevice->queueFamilyIndices.compute;
    loadAssets();
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
    // After Stage 2, final state is in input buffer (readSet=1:
    // particleOut=input) But we need to render from the buffer that contains
    // the latest data Since Stage 2 writes to input when readSet=1, we should
    // render from input However, we need to track which buffer has the latest
    // data For now, let's use input buffer (where Stage 2 writes) Note: We'll
    // fix this properly by tracking the buffer state
    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &storageBuffers.input.buffer,
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

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      compute.pipeline);

    // ========== readSet 逻辑说明 ==========
    // descriptorSet[0]: particleIn=input,  particleOut=output
    // descriptorSet[1]: particleIn=output, particleOut=input  (交换了!)
    //
    // 每一帧的执行流程:
    // 第一帧:
    //   Stage 0: readSet=0 → 从 input 读取，写入 output
    //   Stage 1: readSet=1 → 从 output 读取，写入 input  (切换!)
    //   Stage 2: readSet=1 → 从 output 读取，写入 input
    //   结束后: persistentReadSet = 0 → 下一帧从 input 读取
    //
    // 第二帧:
    //   Stage 0: readSet=0 → 从 input 读取，写入 output
    //   Stage 1: readSet=1 → 从 output 读取，写入 input  (切换!)
    //   Stage 2: readSet=1 → 从 output 读取，写入 input
    //   结束后: persistentReadSet = 0 → 下一帧从 input 读取
    //
    // 关键点:
    //   - Stage 0 不切换 readSet，保持当前值
    //   - Stage 1 每次迭代开始时切换 readSet (ping-pong)
    //   - Stage 2 不切换 readSet，使用 Stage 1 结束后的值
    //   - 每帧结束后，更新 persistentReadSet 为下一帧准备
    // ========================================
    static uint32_t persistentReadSet = 0;
    readSet = persistentReadSet;
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[readSet], 0, 0);

    // Stage 0: Begin solve (computeStage = 0)
    // Initialize predicted positions, save old positions, reset lambda values
    PushConstants pushConsts;
    pushConsts.computeStage = 0;
    pushConsts.parallelSetStartIndex = 1;
    vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                       &pushConsts);

    // Dispatch for all particles
    // 工作组大小是 local_size_x = 10 (1D)
    // 总粒子数 = gridsize.x * gridsize.y = 60 * 60 = 3600
    // 需要的线程数 = 3600
    // 工作组数量 = (3600 + 9) / 10 = 360
    uint32_t numParticles = cloth.gridsize.x * cloth.gridsize.y;
    uint32_t workgroupSizeX = 10; // 匹配 shader 中的 local_size_x
    uint32_t numWorkgroupsX =
        (numParticles + workgroupSizeX - 1) / workgroupSizeX;
    vkCmdDispatch(cmdBuffer, numWorkgroupsX, 1, 1);

    // Barrier after begin solve
    addComputeToComputeBarriers(cmdBuffer, readSet);

    // Stage 1: Constraint solving (computeStage = 1)
    // Iterate over all parallel sets and solve constraints
    const uint32_t numParallelSets =
        static_cast<uint32_t>(compute.elemParallelSlots.size()) - 1;
    const uint32_t constraintIterations = 1;

    for (uint32_t iter = 0; iter < constraintIterations; iter++) {
      // Ping-pong buffers for each iteration
      // 每次迭代开始时切换 readSet: 0→1 或 1→0
      // 这样可以在 input 和 output 之间交替读写，避免数据竞争
      vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              compute.pipelineLayout, 0, 1,
                              &compute.descriptorSets[readSet], 0, 0);

      // Iterate over all parallel sets
      for (uint32_t setIdx = 0; setIdx < numParallelSets; setIdx++) {
        pushConsts.computeStage = 1;
        pushConsts.parallelSetStartIndex = setIdx;
        vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstants), &pushConsts);

        // Dispatch for this parallel set
        // Calculate number of elements in this set
        uint32_t setStart = compute.elemParallelSlots[setIdx];
        uint32_t setEnd = compute.elemParallelSlots[setIdx + 1];
        uint32_t setSize = setEnd - setStart;

        // Dispatch based on element count
        // 工作组大小是 local_size_x = 10 (1D)
        // 工作组数量 = (setSize + 9) / 10
        uint32_t workgroupSizeX = 10; // 匹配 shader 中的 local_size_x
        uint32_t numWorkgroupsX =
            (setSize + workgroupSizeX - 1) / workgroupSizeX;
        vkCmdDispatch(cmdBuffer, numWorkgroupsX, 1, 1);

        // Barrier between parallel sets within same iteration
        if (setIdx < numParallelSets - 1) {
          addComputeToComputeBarriers(cmdBuffer, readSet);
        }
      }

      // Barrier between constraint iterations
      if (iter < constraintIterations - 1) {
        addComputeToComputeBarriers(cmdBuffer, readSet);
      }
    }

    // Stage 2: End solve (computeStage = 2)
    // Update velocities based on position changes
    // After Stage 1, readSet points to the buffer containing predicted
    // positions We need to read from that buffer (particleIn) and write
    // velocities to the other buffer (particleOut) No need to change readSet -
    // keep it as is after Stage 1 After 1 iteration of Stage 1, readSet=1:
    // particleIn=output, particleOut=input

    pushConsts.computeStage = 2;
    pushConsts.parallelSetStartIndex = 0;
    // readSet is already correct from Stage 1, don't override it
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[readSet], 0, 0);
    vkCmdPushConstants(cmdBuffer, compute.pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants),
                       &pushConsts);

    // Dispatch for all particles
    // 工作组大小是 local_size_x = 10 (1D)
    // 总粒子数 = gridsize.x * gridsize.y = 60 * 60 = 3600
    // 需要的线程数 = 3600
    // 工作组数量 = (3600 + 9) / 10 = 360
    uint32_t numParticlesStage2 = cloth.gridsize.x * cloth.gridsize.y;
    uint32_t workgroupSizeXStage2 = 10; // 匹配 shader 中的 local_size_x
    uint32_t numWorkgroupsXStage2 =
        (numParticlesStage2 + workgroupSizeXStage2 - 1) / workgroupSizeXStage2;
    vkCmdDispatch(cmdBuffer, numWorkgroupsXStage2, 1, 1);

    // Release the storage buffers back to the graphics queue
    addComputeToGraphicsBarriers(cmdBuffer, VK_ACCESS_SHADER_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    // After Stage 2, toggle readSet for next frame's Stage 0 to read from
    // correct buffer Stage 2 写入到 input (readSet=1: particleOut=input)
    // 所以下一帧的 Stage 0 应该从 input 读取，即使用 readSet=0
    // (readSet=0: particleIn=input, particleOut=output)
    persistentReadSet = 1 - readSet;

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
    }

    // Debug: Read particle positions and velocities from GPU
    {
      static uint32_t frameCount = 0;
      static uint32_t lastPrintFrame = 0;

      // Wait for compute to complete
      VK_CHECK_RESULT(vkWaitForFences(device, 1, &compute.fences[currentBuffer],
                                      VK_TRUE, UINT64_MAX));

      // Read particle data every 60 frames to avoid too much output
      if (frameCount - lastPrintFrame >= 60) {
        lastPrintFrame = frameCount;

        // Determine which buffer contains the final results after Stage 2
        // After Stage 2, readSet=1, so output buffer contains final positions
        // But actually, after Stage 2, readSet points to the buffer where
        // velocities were written Let's read from output buffer (which should
        // have the final state)
        VkDeviceSize particleBufferSize =
            cloth.gridsize.x * cloth.gridsize.y * sizeof(Particle);

        // Create staging buffer to read from GPU
        vks::Buffer stagingParticleBuffer;
        vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   &stagingParticleBuffer, particleBufferSize);

        // Copy from output buffer (final state) to staging buffer
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(
            VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = particleBufferSize;
        vkCmdCopyBuffer(copyCmd, storageBuffers.output.buffer,
                        stagingParticleBuffer.buffer, 1, &copyRegion);
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

        // Map and read values
        VK_CHECK_RESULT(stagingParticleBuffer.map());
        Particle *particles = (Particle *)stagingParticleBuffer.mapped;

        // Output deltaT and gravity values
        std::cout << "\n========== Frame " << frameCount
                  << " ==========" << std::endl;
        std::cout << "deltaT: " << compute.uniformData.deltaT << std::endl;
        std::cout << "gravity: (" << compute.uniformData.gravity.x << ", "
                  << compute.uniformData.gravity.y << ", "
                  << compute.uniformData.gravity.z << ")" << std::endl;

        // Output positions and velocities for a few key particles
        uint32_t numParticles = cloth.gridsize.x * cloth.gridsize.y;
        uint32_t centerIdx =
            (cloth.gridsize.y / 2) * cloth.gridsize.x + (cloth.gridsize.x / 2);
        uint32_t cornerIdx = 0;              // First particle (top-left)
        uint32_t lastIdx = numParticles - 1; // Last particle

        std::cout << "\nParticle " << cornerIdx << " (first):" << std::endl;
        std::cout << "  pos: (" << particles[cornerIdx].pos.x << ", "
                  << particles[cornerIdx].pos.y << ", "
                  << particles[cornerIdx].pos.z << ")" << std::endl;
        std::cout << "  vel: (" << particles[cornerIdx].vel.x << ", "
                  << particles[cornerIdx].vel.y << ", "
                  << particles[cornerIdx].vel.z << ")" << std::endl;

        if (centerIdx < numParticles) {
          std::cout << "\nParticle " << centerIdx << " (center):" << std::endl;
          std::cout << "  pos: (" << particles[centerIdx].pos.x << ", "
                    << particles[centerIdx].pos.y << ", "
                    << particles[centerIdx].pos.z << ")" << std::endl;
          std::cout << "  vel: (" << particles[centerIdx].vel.x << ", "
                    << particles[centerIdx].vel.y << ", "
                    << particles[centerIdx].vel.z << ")" << std::endl;
        }

        if (lastIdx < numParticles) {
          std::cout << "\nParticle " << lastIdx << " (last):" << std::endl;
          std::cout << "  pos: (" << particles[lastIdx].pos.x << ", "
                    << particles[lastIdx].pos.y << ", "
                    << particles[lastIdx].pos.z << ")" << std::endl;
          std::cout << "  vel: (" << particles[lastIdx].vel.x << ", "
                    << particles[lastIdx].vel.y << ", "
                    << particles[lastIdx].vel.z << ")" << std::endl;
        }

        std::cout << "====================================" << std::endl;
        std::cout.flush();

        stagingParticleBuffer.unmap();
        stagingParticleBuffer.destroy();
      }

      frameCount++;
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
