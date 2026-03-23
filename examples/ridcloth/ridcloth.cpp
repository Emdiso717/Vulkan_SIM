#include <algorithm>
#include <cmath>
#ifdef _WIN32
#pragma comment(linker, "/subsystem:console")
#endif
#include "VulkanglTFModel.h"
#include "vulkanexamplebase.h"
#include <cstdint>

class VulkanExample : public VulkanExampleBase {
public:
  uint32_t indexCount{0};
  bool simulateWind{false};
  bool dedicatedComputeQueue{false};
  float TotalFrameTime = 0.0f;
  float DeltaTime = 1.0 / 120.0f;

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
    glm::uvec2 gridsize{65, 65};
    glm::vec2 size{2.0f, 2.0f};
  } cloth;

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
    struct Pipelines {
      VkPipeline begin{VK_NULL_HANDLE};
      VkPipeline solve{VK_NULL_HANDLE};
      VkPipeline end{VK_NULL_HANDLE};
    } pipelines;
    struct UniformData {
      float deltaT{0.0f};
      float particleMass{0.19f};
      float springStiffness{16778.523490};
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
    std::vector<Particle> particleBuffer(cloth.gridsize.x * cloth.gridsize.y);

    float dx = cloth.size.x / (cloth.gridsize.x - 1);
    float dy = cloth.size.y / (cloth.gridsize.y - 1);
    float du = 1.0f / (cloth.gridsize.x - 1);
    float dv = 1.0f / (cloth.gridsize.y - 1);

    // Set up a slightly tilted cloth that falls onto sphere
    glm::mat4 transM =
        glm::translate(glm::mat4(1.0f), glm::vec3(-cloth.size.x / 2.0f, -2.0f,
                                                  -cloth.size.y / 2.0f));
    // Small initial tilt to break perfect symmetry
    transM =
        glm::rotate(transM, glm::radians(30.0f), glm::vec3(0.0f, 0.0f, 1.0f));
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
    vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.input.buffer,
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
    // Lambda buffer
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
    copyRegion.size = elemParallelSlotsBufferSize;
    vkCmdCopyBuffer(copyCmd, stagingElemParallelSlotsBuffer.buffer,
                    compute.elemParallelSlotsBuffer.buffer, 1, &copyRegion);
    vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
    // Clean up staging buffers
    stagingElementInfoBuffer.destroy();
    stagingLambdaBuffer.destroy();
    stagingElemParallelSlotsBuffer.destroy();
  }

  void prepareDescriptorPool() {
    // This is shared between graphics and compute
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                              maxConcurrentFrames * 3),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              maxConcurrentFrames * 7),
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
        loadShader(getShadersPath() + "ridcloth/cloth.vert.spv",
                   VK_SHADER_STAGE_VERTEX_BIT),
        loadShader(getShadersPath() + "ridcloth/cloth.frag.spv",
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
    shaderStages = {loadShader(getShadersPath() + "ridcloth/sphere.vert.spv",
                               VK_SHADER_STAGE_VERTEX_BIT),
                    loadShader(getShadersPath() + "ridcloth/sphere.frag.spv",
                               VK_SHADER_STAGE_FRAGMENT_BIT)};
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1,
                                              &pipelineCreateInfo, nullptr,
                                              &graphics.pipelines.sphere));
  }

  void prepareComputeParallel() {
    // Set some initial values
    float dx = cloth.size.x / (cloth.gridsize.x - 1);
    float dy = cloth.size.y / (cloth.gridsize.y - 1);

    compute.uniformData.restDistH = dx;
    compute.uniformData.restDistV = dy;
    compute.uniformData.restDistD = sqrtf(dx * dx + dy * dy);
    compute.uniformData.particleCount = cloth.gridsize;
    int elemId = 0;
    for (uint32_t y = 0; y < cloth.gridsize.y; y++) {
      for (uint32_t x = 0; x < cloth.gridsize.x; x++) {
        int currentIdx = y + x * cloth.gridsize.y;
        if (x < cloth.gridsize.x - 1) {
          int rightIdx = y + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, rightIdx);
          elem.restLength = compute.uniformData.restDistH;
          compute.elementInfo.push_back(elem);
        }
        if (y < cloth.gridsize.y - 1) {
          int bottomIdx = (y + 1) + x * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomIdx);
          elem.restLength = compute.uniformData.restDistV;
          compute.elementInfo.push_back(elem);
        }
        if (x < cloth.gridsize.x - 1 && y < cloth.gridsize.y - 1) {
          int bottomRightIdx = (y + 1) + (x + 1) * cloth.gridsize.y;
          ElementInfo elem;
          elem.elemId = elemId++;
          elem.pid = glm::ivec2(currentIdx, bottomRightIdx);
          elem.restLength = compute.uniformData.restDistD;
          compute.elementInfo.push_back(elem);
        }
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
            &compute.elemParallelSlotsBuffer.descriptor)};

    vkUpdateDescriptorSets(
        device, static_cast<uint32_t>(computeWriteDescriptorSets.size()),
        computeWriteDescriptorSets.data(), 0, NULL);

    // Create pipelines (begin / solve / end)
    VkComputePipelineCreateInfo computePipelineCreateInfo =
        vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "ridcloth/cloth_begin.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.begin));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "ridcloth/cloth_solve.comp.spv",
                   VK_SHADER_STAGE_COMPUTE_BIT);
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, pipelineCache, 1, &computePipelineCreateInfo, nullptr,
        &compute.pipelines.solve));

    computePipelineCreateInfo.stage =
        loadShader(getShadersPath() + "ridcloth/cloth_end.comp.spv",
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
      compute.uniformData.deltaT = DeltaTime;
      // fmin(frameTimer, 0.02) * 0.8;
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

    // // Render sphere
    // vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                   graphics.pipelines.sphere);
    // vkCmdBindDescriptorSets(
    //     cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout,
    //     0, 1, &graphics.descriptorSets[currentBuffer], 0, nullptr);
    // modelSphere.draw(cmdBuffer);

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

    // release the storage buffers to the compute queue
    addGraphicsToComputeBarriers(cmdBuffer,
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
  }

  void buildComputeCommandBuffer(uint32_t substeps) {
    VkCommandBuffer cmdBuffer = compute.commandBuffers[currentBuffer];

    VkCommandBufferBeginInfo cmdBufInfo =
        vks::initializers::commandBufferBeginInfo();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

    // Acquire the storage buffers from the graphics queue
    addGraphicsToComputeBarriers(cmdBuffer, 0, VK_ACCESS_SHADER_READ_BIT,
                                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // If we don't need to advance simulation this frame, we still need to
    // transfer ownership back to graphics (graphics command buffer will
    // acquire).
    if (substeps == 0) {
      addComputeToGraphicsBarriers(cmdBuffer, 0, 0,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
      vkEndCommandBuffer(cmdBuffer);
      return;
    }

    // Single descriptor set, fixed binding of input/output buffers
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compute.pipelineLayout, 0, 1,
                            &compute.descriptorSets[0], 0, 0);

    const uint32_t numParticles = cloth.gridsize.x * cloth.gridsize.y;
    const uint32_t numLambda = static_cast<uint32_t>(compute.lambdaData.size());
    const uint32_t workgroupSizeX = 64;
    const uint32_t numWorkgroupsX =
        (std::max(numParticles, numLambda) + workgroupSizeX - 1) /
        workgroupSizeX;

    const uint32_t numParallelSets =
        static_cast<uint32_t>(compute.elemParallelSlots.size()) - 1;
    const uint32_t constraintIterations = 15;

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
      addComputeToComputeBarriers(cmdBuffer);

      // Stage 1: Constraint solving
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.solve);

      for (uint32_t iter = 0; iter < constraintIterations; iter++) {
        // Iterate over all parallel sets
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
      vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        compute.pipelines.end);
      vkCmdDispatch(cmdBuffer, numWorkgroupsXStage2, 1, 1);

      // Barrier between substeps (except after the last one)
      if (step + 1 < substeps) {
        addComputeToComputeBarriers(cmdBuffer);
      }
    }

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

      TotalFrameTime += frameTimer;
      updateComputeUBO();
      const uint32_t maxSubstepsPerFrame = 16;
      uint32_t substeps =
          static_cast<uint32_t>(std::floor(TotalFrameTime / DeltaTime));
      if (substeps > maxSubstepsPerFrame) {
        substeps = maxSubstepsPerFrame;
      }
      TotalFrameTime -= static_cast<float>(substeps) * DeltaTime;
      // Avoid unbounded catch-up if we fall behind
      if (TotalFrameTime >
          DeltaTime * static_cast<float>(maxSubstepsPerFrame)) {
        TotalFrameTime = DeltaTime * static_cast<float>(maxSubstepsPerFrame);
      }
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

  virtual void writeMesh() { int k; }
};

VULKAN_EXAMPLE_MAIN()
