#pragma once

#include <cstdint>

// number of tokens per asynchronous block
inline constexpr int64_t ACS_TokensPerBlock = 64ll;
// upper bound for bytes staged in a single block copy (can be overridden via env)
inline constexpr int64_t ACS_DefaultMaxBlockSizeBytes = 32ll * 1024ll * 1024ll;
inline constexpr const char *ACS_MaxBlockSizeEnvVar = "KVCACHE_MAX_BLOCK_SIZE";
// for control SM usage
inline constexpr int64_t ACS_KernelBlocks = 64ll;
// task queue length
inline constexpr int64_t ACS_QueueSize = 131072ll;


