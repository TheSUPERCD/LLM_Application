#pragma once

#include <llama.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace TermColor {
  inline constexpr const char* NORMAL   = "\033[0m";
  inline constexpr const char* RED      = "\033[1;31m";
  inline constexpr const char* GREEN    = "\033[1;32m";
  inline constexpr const char* YELLOW   = "\033[1;33m";
  inline constexpr const char* BLUE     = "\033[1;34m";
  inline constexpr const char* MAGENTA  = "\033[1;35m";
  inline constexpr const char* CYAN     = "\033[1;36m";
  inline constexpr const char* CLR_LINE = "\r\033[2K";
}

struct ChatMessage {
  std::string role;
  std::string content;
};

struct DeviceInfo {
  size_t index = 0;
  std::string name, description, type;
  size_t free_memory_bytes = 0, total_memory_bytes = 0;
};

enum class KVCacheType { AUTO = 0, F16 = 1, Q8_0 = 2, Q4_0 = 3 };

struct EngineConfig {
  std::string model_path;
  std::string mmproj_path;
  uint32_t context_size = 4096, batch_size = 512, ubatch_size = 0;
  int32_t gpu_layers = 99, device_index = -1, threads = 0, threads_batch = 0;
  std::string device_name;
  bool flash_attn = true, offload_kqv = true, disable_thinking = false;
  bool use_mmap = true, use_mlock = false, no_host = false;
  bool safe_mode = false, force_performance = false;
  KVCacheType kv_cache_type = KVCacheType::AUTO;
  std::string log_file = "";
  bool verbose_logging = false;
  double watchdog_budget_sec = 2.0;
};

struct FilterConfig {
  bool hide_thinking = false;
  std::string think_start_tag = "<think>";
  std::string think_end_tag = "</think>";
  std::vector<std::string> stop_words = {
    "<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end|>", "</s>"
  };
};

struct InferenceParams : FilterConfig {
  float temperature = 0.8f, top_p = 0.95f, min_p = 0.05f, repeat_penalty = 1.1f, presence_penalty = 0.0f;
  int32_t top_k = 40, repeat_last_n = 64, max_tokens = -1;
  uint32_t seed = LLAMA_DEFAULT_SEED;
};

struct GenerationStats {
  int prompt_tokens = 0, generated_tokens = 0, thinking_tokens = 0;
  double prompt_eval_time_sec = 0.0, generation_time_sec = 0.0, thinking_time_sec = 0.0;
  double prompt_tps = 0.0, generation_tps = 0.0;
};

using TokenCallback = std::function<bool(std::string_view piece)>;
using ThinkSpinnerCallback = std::function<void(int think_tokens, double elapsed_sec)>;
using ProgressCallback = std::function<bool(float progress)>;

// Integrated streaming filter for reasoning models with <think> tag handling and stop words
class ThinkingStreamFilter {
public:
  using Config = FilterConfig;
  using TextEmitter = std::function<void(std::string_view text)>;

  explicit ThinkingStreamFilter(const Config& config,
                                TextEmitter emitter,
                                ThinkSpinnerCallback spinner_cb = nullptr,
                                TextEmitter ui_emitter = nullptr);

  bool ProcessPiece(std::string_view piece);
  void Flush();

  int GetThinkingTokens() const { return think_tokens_; }
  double GetThinkingDurationSec() const;

private:
  void process_buffer();
  void emit_ui(std::string_view text);

  Config config_;
  TextEmitter emitter_;
  ThinkSpinnerCallback spinner_cb_;
  TextEmitter ui_emitter_;

  bool in_thinking_ = false;
  bool hit_stop_word_ = false;
  int think_tokens_ = 0;
  double thinking_duration_sec_ = 0.0;
  std::chrono::steady_clock::time_point think_start_time_;
  std::string buffer_;
};

class LLMEngine {
public:
  explicit LLMEngine(const EngineConfig& config, ProgressCallback progress_cb = nullptr);
  ~LLMEngine();

  LLMEngine(const LLMEngine&) = delete;
  LLMEngine& operator=(const LLMEngine&) = delete;
  LLMEngine(LLMEngine&&) noexcept;
  LLMEngine& operator=(LLMEngine&&) noexcept;

  bool ProcessPrompt(const std::string& prompt, GenerationStats* stats = nullptr);
  bool ProcessPrompt(const std::string& prompt,
                     const std::vector<std::string>& media_paths,
                     GenerationStats* stats = nullptr);
  std::string Generate(const InferenceParams& params,
                       TokenCallback token_cb = nullptr,
                       GenerationStats* stats = nullptr,
                       ThinkSpinnerCallback spinner_cb = nullptr);

  void ResetContext();
  uint32_t GetContextUsage() const;
  uint32_t GetContextSize() const;

  bool LoadMultimodalProjector(const std::string& mmproj_path);
  bool HasMultimodal() const;
  bool SupportsVision() const;
  bool SupportsAudio() const;

  const char *GetChatTemplate() const;
  std::string ApplyChatTemplate(const std::vector<ChatMessage>& history, bool add_generation_prompt = true) const;
  size_t PruneChatHistory(std::vector<ChatMessage>& history, size_t target_tokens) const;
  std::string GetModelDescription() const;
  std::string GetQuantizationType() const;
  std::string GetKVCacheTypeShortName() const;
  int GetOffloadedLayers() const;
  uint32_t GetTotalLayers() const;
  static std::vector<DeviceInfo> GetAvailableDevices();

  void AbortGeneration();
  size_t CountTokens(const std::string& text) const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};
