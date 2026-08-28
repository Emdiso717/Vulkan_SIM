#pragma once

// Reusable synchronization helpers for examples that share buffers between
// graphics and compute queues.

#include "VulkanInitializers.hpp"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace example_barriers {

inline void addBufferBarriers(
    VkCommandBuffer commandBuffer, std::initializer_list<VkBuffer> buffers,
    VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask,
    VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
    uint32_t srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    uint32_t dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED) {
  if (buffers.size() == 0) {
    return;
  }

  std::vector<VkBufferMemoryBarrier> barriers;
  barriers.reserve(buffers.size());
  for (VkBuffer buffer : buffers) {
    VkBufferMemoryBarrier barrier = vks::initializers::bufferMemoryBarrier();
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = srcQueueFamilyIndex;
    barrier.dstQueueFamilyIndex = dstQueueFamilyIndex;
    barrier.buffer = buffer;
    barrier.size = VK_WHOLE_SIZE;
    barriers.push_back(barrier);
  }

  vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, 0, 0, nullptr,
                       static_cast<uint32_t>(barriers.size()), barriers.data(),
                       0, nullptr);
}

inline void addGraphicsToComputeBarriers(
    VkCommandBuffer commandBuffer, std::initializer_list<VkBuffer> buffers,
    bool dedicatedComputeQueue, uint32_t graphicsQueueFamily,
    uint32_t computeQueueFamily, VkAccessFlags srcAccessMask,
    VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask) {
  if (!dedicatedComputeQueue) {
    return;
  }
  addBufferBarriers(commandBuffer, buffers, srcAccessMask, dstAccessMask,
                    srcStageMask, dstStageMask, graphicsQueueFamily,
                    computeQueueFamily);
}

inline void
addComputeToComputeBarriers(VkCommandBuffer commandBuffer,
                            std::initializer_list<VkBuffer> buffers) {
  addBufferBarriers(commandBuffer, buffers, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

inline void addComputeToGraphicsBarriers(
    VkCommandBuffer commandBuffer, std::initializer_list<VkBuffer> buffers,
    bool dedicatedComputeQueue, uint32_t computeQueueFamily,
    uint32_t graphicsQueueFamily, VkAccessFlags srcAccessMask,
    VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask) {
  if (!dedicatedComputeQueue) {
    return;
  }
  addBufferBarriers(commandBuffer, buffers, srcAccessMask, dstAccessMask,
                    srcStageMask, dstStageMask, computeQueueFamily,
                    graphicsQueueFamily);
}

} // namespace example_barriers
