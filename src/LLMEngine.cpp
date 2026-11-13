#include <LLMEngine.hpp>
#include <gguf.h>
#include <mtmd.h>
#include <mtmd-helper.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef _WIN32
static inline int set_env(const char* name, const char* value, int overwrite) {
  if (overwrite || !getenv(name)) return _putenv_s(name, value);
  return 0;
}
static inline int unset_env(const char* name) {
  return _putenv_s(name, "");
}
#else
static inline int set_env(const char* name, const char* value, int overwrite) {
  return setenv(name, value, overwrite);
}
static inline int unset_env(const char* name) {
  return unsetenv(name);
}
#endif

static size_t total_system_ram_bytes() {
#ifdef _WIN32
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  return GlobalMemoryStatusEx(&status) ? static_cast<size_t>(status.ullTotalPhys) : 0;
#else
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages <= 0 || page_size <= 0) return 0;
  return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
#endif
}

namespace {

struct GGUFOffloadProfile {
  uint32_t total_layers = 0, context_length = 0, n_head_kv = 0, head_dim = 0;
  size_t layer_weight_bytes = 0, base_weight_bytes = 0, total_model_bytes = 0;
  std::string quantization_type = "Unknown";
  float avg_bits_per_weight = 16.0f;
  bool is_quantized = false;
};

// Single source of truth for the KV cache variants (indexed by KVCacheType).
struct KVCacheInfo {
  double bytes_per_elem;
  const char* short_name;
};

static const KVCacheInfo& get_kv_cache_info(KVCacheType kv_type) {
  static const KVCacheInfo info[] = {
    /* AUTO */ { 2.0,    "F16"  },
    /* F16  */ { 2.0,    "F16"  },
    /* Q8_0 */ { 1.0625, "Q8_0" },
    /* Q4_0 */ { 0.5625, "Q4_0" },
  };
  const size_t idx = static_cast<size_t>(kv_type);
  return info[idx < sizeof(info) / sizeof(info[0]) ? idx : 0];
}

static KVCacheType resolve_optimal_kv_cache(KVCacheType requested, const GGUFOffloadProfile& profile) {
  if (requested != KVCacheType::AUTO) return requested;
  const std::string& q = profile.quantization_type;
  for (const char* pfx : {"Q4", "IQ4", "Q3", "IQ3", "Q2", "IQ2", "Q1", "IQ1", "TQ", "FP4"}) {
    if (q.find(pfx) != std::string::npos) return KVCacheType::Q4_0;
  }
  for (const char* pfx : {"Q5", "Q6", "Q8", "IQ5"}) {
    if (q.find(pfx) != std::string::npos) return KVCacheType::Q8_0;
  }
  if (profile.is_quantized) {
    if (profile.avg_bits_per_weight <= 4.7f) return KVCacheType::Q4_0;
    if (profile.avg_bits_per_weight <= 8.5f) return KVCacheType::Q8_0;
  }
  return KVCacheType::F16;
}

static GGUFOffloadProfile probe_model_offload_profile(const std::string& model_path) {
  GGUFOffloadProfile p;
  struct gguf_init_params params = { true, nullptr };
  struct gguf_context* ctx = gguf_init_from_file(model_path.c_str(), params);
  if (!ctx) return p;

  int64_t aid = gguf_find_key(ctx, "general.architecture");
  if (aid >= 0) {
    std::string arch = gguf_get_val_str(ctx, aid);
    auto get_u32 = [&](const std::string& k, uint32_t def = 0) -> uint32_t {
      int64_t id = gguf_find_key(ctx, (arch + k).c_str());
      return (id >= 0) ? gguf_get_val_u32(ctx, id) : def;
    };
    p.total_layers = get_u32(".block_count");
    p.context_length = get_u32(".context_length", 4096);
    p.n_head_kv = get_u32(".attention.head_count_kv", get_u32(".attention.head_count", 8));
    p.head_dim = get_u32(".attention.key_length");
    if (p.head_dim == 0) {
      uint32_t embd = get_u32(".embedding_length"), heads = get_u32(".attention.head_count", 1);
      p.head_dim = (heads > 0) ? (embd / heads) : 128;
    }
  }

  int64_t ft_id = gguf_find_key(ctx, "general.file_type");
  if (ft_id >= 0) {
    const char* ftype_name = llama_ftype_name(static_cast<enum llama_ftype>(gguf_get_val_u32(ctx, ft_id)));
    if (ftype_name && std::strlen(ftype_name) > 0) p.quantization_type = ftype_name;
  }

  std::vector<size_t> type_bytes(GGML_TYPE_COUNT, 0);
  size_t layer0_bytes = 0, output_bytes = 0;
  int64_t n_tensors = gguf_get_n_tensors(ctx);
  for (int64_t i = 0; i < n_tensors; ++i) {
    size_t sz = gguf_get_tensor_size(ctx, i);
    p.total_model_bytes += sz;
    enum ggml_type t = gguf_get_tensor_type(ctx, i);
    if (t < GGML_TYPE_COUNT) type_bytes[t] += sz;

    const char* tname = gguf_get_tensor_name(ctx, i);
    if (tname) {
      if (std::strstr(tname, "blk.0.")) layer0_bytes += sz;
      else if (std::strncmp(tname, "output", 6) == 0) output_bytes += sz;
    }
  }

  size_t max_bytes = 0;
  enum ggml_type dominant_type = GGML_TYPE_F16;
  for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
    if (type_bytes[i] > max_bytes) {
      max_bytes = type_bytes[i];
      dominant_type = static_cast<enum ggml_type>(i);
    }
  }
  p.is_quantized = ggml_is_quantized(dominant_type);
  p.avg_bits_per_weight = static_cast<float>((static_cast<double>(ggml_type_size(dominant_type)) / static_cast<double>(ggml_blck_size(dominant_type))) * 8.0);

  if (p.quantization_type == "Unknown" || p.quantization_type.empty()) {
    p.quantization_type = ggml_type_name(dominant_type);
  }

  p.layer_weight_bytes = (layer0_bytes > 0) ? layer0_bytes :
      ((p.total_layers > 0 && p.total_model_bytes > output_bytes)
       ? (p.total_model_bytes - output_bytes) / p.total_layers : 0);
  p.base_weight_bytes = output_bytes;
  gguf_free(ctx);
  return p;
}

static int calculate_optimal_gpu_layers(const GGUFOffloadProfile& profile, size_t free_vram,
                                        const EngineConfig& config, KVCacheType resolved_kv) {
  if (profile.total_layers == 0 || free_vram == 0) return config.gpu_layers;

  uint32_t effective_ctx = config.context_size ? config.context_size : (profile.context_length ? profile.context_length : 4096);
  uint32_t effective_ubatch = std::min(config.ubatch_size, effective_ctx);
  size_t compute_buf = std::min<size_t>(512ULL * 1024ULL * 1024ULL, static_cast<size_t>(effective_ubatch) * 1024ULL * 1024ULL);
  const size_t overhead_reserve = (384ULL * 1024ULL * 1024ULL) + compute_buf;
  if (free_vram <= overhead_reserve) return 0;

  double bpe = get_kv_cache_info(resolved_kv).bytes_per_elem;
  size_t kv_per_layer = config.offload_kqv ? static_cast<size_t>(effective_ctx * 2.0 * profile.n_head_kv * profile.head_dim * bpe) : 0;

  int32_t max_offloadable = static_cast<int32_t>(profile.total_layers);
  size_t total_needed = profile.total_model_bytes + (profile.total_layers * kv_per_layer) + overhead_reserve;
  if (free_vram >= total_needed) return std::min(config.gpu_layers, max_offloadable);

  size_t budget = free_vram - overhead_reserve;
  if (budget <= profile.base_weight_bytes) return 0;
  budget -= profile.base_weight_bytes;

  size_t mem_per_layer = profile.layer_weight_bytes + kv_per_layer;
  if (mem_per_layer == 0) return config.gpu_layers;

  int safe_layers = static_cast<int>(budget / mem_per_layer);
  return std::min(config.gpu_layers, std::min(max_offloadable, std::max(0, safe_layers)));
}

} // namespace

// Thread-safe Log Manager using llama.cpp callback API
class LogManager {
public:
  static LogManager& instance() { static LogManager inst; return inst; }

  // Install the callbacks without opening a log file or enabling stderr output.
  void ensure_silent() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (callbacks_installed_) return;
    llama_log_set(log_cb, this);
    mtmd_helper_log_set(log_cb, this);
    callbacks_installed_ = true;
  }

  void setup(const std::string& log_file, bool verbose) {
    std::lock_guard<std::mutex> lock(mutex_);
    verbose_ = verbose;
    if (log_fp_) { fclose(log_fp_); log_fp_ = nullptr; }
    if (!log_file.empty()) {
      log_fp_ = fopen(log_file.c_str(), "a");
      if (log_fp_) {
        auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char s[32];
        std::strftime(s, sizeof(s), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        std::fprintf(log_fp_, "\n\n=== LLM Engine Started at %s ===\n", s);
        std::fflush(log_fp_);
      }
    }
    llama_log_set(log_cb, this);
    mtmd_helper_log_set(log_cb, this);
    callbacks_installed_ = true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    llama_log_set(nullptr, nullptr);
    mtmd_helper_log_set(nullptr, nullptr);
    callbacks_installed_ = false;
    if (log_fp_) { fclose(log_fp_); log_fp_ = nullptr; }
  }

  ~LogManager() { close(); }

private:
  FILE* log_fp_ = nullptr;
  bool verbose_ = false;
  bool callbacks_installed_ = false;
  std::mutex mutex_;

  static void log_cb(ggml_log_level, const char* text, void* user) {
    if (!text) return;
    auto* self = static_cast<LogManager*>(user);
    std::lock_guard<std::mutex> lock(self->mutex_);
    if (self->log_fp_) { std::fputs(text, self->log_fp_); std::fflush(self->log_fp_); }
    if (self->verbose_) { std::fputs(text, stderr); std::fflush(stderr); }
  }
};

static void ensure_backend_init() {
  static struct BackendInit {
    BackendInit() {
      set_env("GGML_VK_ALLOW_GRAPHICS_QUEUE", "1", 1);
      llama_backend_init();
      llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    }
    ~BackendInit() { llama_backend_free(); }
  } init;
  (void)init;
}

// RAII Deleters
struct LlamaModelDeleter { void operator()(llama_model* m) const { llama_model_free(m); } };
struct LlamaContextDeleter { void operator()(llama_context* c) const { llama_free(c); } };
struct LlamaSamplerDeleter { void operator()(llama_sampler* s) const { llama_sampler_free(s); } };

// Scoped Batch RAII wrapper aligned to 64-byte CPU cache line boundary
struct alignas(64) ScopedBatch {
  llama_batch batch = {};
  int32_t capacity = 0;

  ScopedBatch() = default;
  ~ScopedBatch() { reset(); }

  void init(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1) {
    if (capacity >= n_tokens && batch.token) {
      batch.n_tokens = 0;
      return;
    }
    reset();
    batch = llama_batch_init(n_tokens, embd, n_seq_max);
    capacity = n_tokens;
  }

  void reset() {
    if (batch.token || batch.embd) {
      llama_batch_free(batch);
      batch = {};
      capacity = 0;
    }
  }

  ScopedBatch(const ScopedBatch&) = delete;
  ScopedBatch& operator=(const ScopedBatch&) = delete;
  llama_batch* operator->() { return &batch; }
  llama_batch& operator*() { return batch; }
};

static size_t suffix_match(std::string_view buf, std::string_view pat) {
  if (pat.empty()) return 0;
  for (size_t len = std::min(buf.length(), pat.length() - 1); len >= 1; --len) {
    if (buf.compare(buf.length() - len, len, pat, 0, len) == 0) return len;
  }
  return 0;
}

static size_t max_suffix_match(std::string_view buf, std::string_view tag, const std::vector<std::string>& stop_words) {
  size_t m = suffix_match(buf, tag);
  for (const auto& sw : stop_words) {
    if (!sw.empty()) m = std::max(m, suffix_match(buf, sw));
  }
  return m;
}

static std::string format_thought_banner(double sec, bool hide_thinking, bool force_1dec = false) {
  char buf[96];
  if (force_1dec || hide_thinking) {
    std::snprintf(buf, sizeof(buf), "%s%s[Thought for %.1fs]%s\n%s",
                  TermColor::CLR_LINE, TermColor::CYAN, sec, TermColor::NORMAL, TermColor::GREEN);
  } else {
    std::snprintf(buf, sizeof(buf), "\n%s[Thought for %.2fs]%s\n%s",
                  TermColor::CYAN, sec, TermColor::NORMAL, TermColor::GREEN);
  }
  return buf;
}

// ThinkingStreamFilter Implementation
ThinkingStreamFilter::ThinkingStreamFilter(const Config& config, TextEmitter emitter,
                                           ThinkSpinnerCallback spinner_cb, TextEmitter ui_emitter)
    : config_(config), emitter_(std::move(emitter)), spinner_cb_(std::move(spinner_cb)),
      ui_emitter_(std::move(ui_emitter)) {
  buffer_.reserve(512);
}

void ThinkingStreamFilter::emit_ui(std::string_view text) {
  if (ui_emitter_) ui_emitter_(text);
  else if (emitter_) emitter_(text);
}

double ThinkingStreamFilter::GetThinkingDurationSec() const {
  if (in_thinking_) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - think_start_time_).count();
  }
  return thinking_duration_sec_;
}

bool ThinkingStreamFilter::ProcessPiece(std::string_view piece) {
  if (hit_stop_word_) return false;
  buffer_.append(piece.data(), piece.size());
  process_buffer();
  return !hit_stop_word_;
}

void ThinkingStreamFilter::process_buffer() {
  const std::string& start_tag = config_.think_start_tag;
  const std::string& end_tag = config_.think_end_tag;

  while (!buffer_.empty() && !hit_stop_word_) {
    for (const auto& sw : config_.stop_words) {
      if (sw.empty()) continue;
      size_t pos = buffer_.find(sw);
      if (pos != std::string::npos) {
        if (pos > 0 && emitter_ && !in_thinking_) emitter_(std::string_view(buffer_).substr(0, pos));
        buffer_.clear();
        hit_stop_word_ = true;
        in_thinking_ = false;
        return;
      }
    }

    if (!in_thinking_) {
      size_t pos = buffer_.find(start_tag);
      if (pos != std::string::npos) {
        if (pos > 0 && emitter_) emitter_(std::string_view(buffer_).substr(0, pos));
        in_thinking_ = true;
        think_tokens_ = 0;
        think_start_time_ = std::chrono::steady_clock::now();
        if (!config_.hide_thinking) {
          emit_ui(TermColor::CYAN);
          if (emitter_) emitter_(start_tag);
        }
        buffer_.erase(0, pos + start_tag.length());
        continue;
      }

      size_t longest = max_suffix_match(buffer_, start_tag, config_.stop_words);
      if (longest > 0) {
        size_t safe = buffer_.length() - longest;
        if (safe > 0 && emitter_) emitter_(std::string_view(buffer_).substr(0, safe));
        buffer_.erase(0, safe);
        break;
      }
      if (emitter_) emitter_(buffer_);
      buffer_.clear();
    } else {
      size_t pos = buffer_.find(end_tag);
      if (pos != std::string::npos) {
        thinking_duration_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - think_start_time_).count();
        if (!config_.hide_thinking && emitter_) emitter_(buffer_.substr(0, pos) + end_tag);
        emit_ui(format_thought_banner(thinking_duration_sec_, config_.hide_thinking));
        in_thinking_ = false;
        buffer_.erase(0, pos + end_tag.length());
        continue;
      }

      size_t match_len = max_suffix_match(buffer_, end_tag, config_.stop_words);
      if (match_len > 0) {
        size_t safe = buffer_.length() - match_len;
        if (safe > 0 && !config_.hide_thinking && emitter_) emitter_(std::string_view(buffer_).substr(0, safe));
        buffer_.erase(0, safe);
        break;
      }

      think_tokens_++;
      if (!config_.hide_thinking && emitter_) {
        emitter_(buffer_);
      } else if (config_.hide_thinking && spinner_cb_) {
        std::chrono::duration<double> cur = std::chrono::steady_clock::now() - think_start_time_;
        spinner_cb_(think_tokens_, cur.count());
      }
      buffer_.clear();
    }
  }
}

void ThinkingStreamFilter::Flush() {
  if (hit_stop_word_) { buffer_.clear(); return; }
  if (in_thinking_) {
    thinking_duration_sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - think_start_time_).count();
    if (config_.hide_thinking) emit_ui(format_thought_banner(thinking_duration_sec_, true, true));
    in_thinking_ = false;
  }
  for (const auto& sw : config_.stop_words) {
    if (sw.empty()) continue;
    size_t pos = buffer_.find(sw);
    if (pos != std::string::npos) buffer_.erase(pos);
    size_t m = suffix_match(buffer_, sw);
    if (m > 0) buffer_.erase(buffer_.length() - m);
  }
  if (!buffer_.empty() && emitter_) emitter_(buffer_);
  buffer_.clear();
}

// Private implementation of LLMEngine
struct alignas(64) LLMEngine::Impl {
  EngineConfig config;
  std::unique_ptr<llama_model, LlamaModelDeleter> model;
  const llama_vocab* vocab = nullptr;
  std::unique_ptr<llama_context, LlamaContextDeleter> ctx;
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> smpl;
  ScopedBatch batch;
  uint32_t n_past = 0;
  bool supports_disabling_thinking = false;
  llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;
  int32_t n_swa = 0;
  std::atomic<bool> abort_requested{false};
  alignas(64) mutable std::vector<llama_token> scratch_tokens_;
  ggml_backend_dev_t selected_devices_[2] = {nullptr, nullptr};

  std::string quantization_type = "Unknown";
  KVCacheType resolved_kv_cache_type = KVCacheType::F16;
  int offloaded_layers = 0;
  uint32_t total_layers = 0;
  mtmd::context_ptr ctx_mtmd;
  bool safe_mode = false;

  Impl(const EngineConfig& cfg, ProgressCallback progress_cb) : config(cfg) {
    GGUFOffloadProfile profile = probe_model_offload_profile(config.model_path);

    // Large models can exhaust RADV command-submission memory during upload on
    // UMA. Fall back to the conservative path only when the model is close to
    // the machine's total RAM.
    const size_t system_ram = total_system_ram_bytes();
    safe_mode = config.safe_mode ||
                (!config.force_performance && system_ram > 0 &&
                 profile.total_model_bytes > (system_ram * 3) / 4);

    if (safe_mode) {
      set_env("GGML_VK_SERIALIZE_SUBMISSIONS", "1", 1);
      set_env("GGML_VK_MAX_NODES_PER_SUBMIT", "16", 1);
      set_env("GGML_VK_DISABLE_ASYNC", "1", 1);
    } else {
      unset_env("GGML_VK_SERIALIZE_SUBMISSIONS");
      unset_env("GGML_VK_MAX_NODES_PER_SUBMIT");
      unset_env("GGML_VK_DISABLE_ASYNC");
    }
    if (config.watchdog_budget_sec > 0.0) {
      char b_buf[32];
      std::snprintf(b_buf, sizeof(b_buf), "%.2f", config.watchdog_budget_sec);
      set_env("GGML_VK_WATCHDOG_BUDGET_SEC", b_buf, 1);
    }
    LogManager::instance().setup(config.log_file, config.verbose_logging);
    ensure_backend_init();

    if (config.context_size == 0) {
      config.context_size = (profile.context_length > 0) ? profile.context_length : 4096;
    }
    if (config.ubatch_size == 0) {
      config.ubatch_size = profile.total_model_bytes > (12ULL << 30) ? 256 : 128;
    }

    quantization_type = profile.quantization_type;
    total_layers = profile.total_layers;
    resolved_kv_cache_type = resolve_optimal_kv_cache(config.kv_cache_type, profile);

    llama_model_params mp = llama_model_default_params();
    if (config.gpu_layers > 0) {
      mp.n_gpu_layers = config.gpu_layers;
      if (!config.device_name.empty()) {
        selected_devices_[0] = ggml_backend_dev_by_name(config.device_name.c_str());
      } else if (config.device_index >= 0 && static_cast<size_t>(config.device_index) < ggml_backend_dev_count()) {
        selected_devices_[0] = ggml_backend_dev_get(config.device_index);
      } else {
        selected_devices_[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!selected_devices_[0]) selected_devices_[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
      }
    } else {
      mp.n_gpu_layers = 0;
      selected_devices_[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (selected_devices_[0]) mp.devices = selected_devices_;

    bool is_uma = false;
    if (selected_devices_[0]) {
      ggml_backend_dev_props dev_props;
      ggml_backend_dev_get_props(selected_devices_[0], &dev_props);
      is_uma = (dev_props.type == GGML_BACKEND_DEVICE_TYPE_IGPU);
    }

    if (safe_mode || is_uma) {
      set_env("GGML_VK_FLASH_ATTN_F32ACC", "1", 1);
    } else {
      unset_env("GGML_VK_FLASH_ATTN_F32ACC");
    }

    mp.no_host = is_uma || config.no_host;
    mp.load_mode = !config.use_mmap ? (config.use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE) :
                   (config.use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_AUTO);

    if (config.gpu_layers > 0 && selected_devices_[0] &&
        ggml_backend_dev_type(selected_devices_[0]) != GGML_BACKEND_DEVICE_TYPE_CPU) {
      size_t free_vram = 0, total_vram = 0;
      ggml_backend_dev_memory(selected_devices_[0], &free_vram, &total_vram);
      if (free_vram > 0) {
        if (config.kv_cache_type == KVCacheType::AUTO && resolved_kv_cache_type == KVCacheType::Q8_0) {
          int layers_q8 = calculate_optimal_gpu_layers(profile, free_vram, config, KVCacheType::Q8_0);
          int max_offloadable = static_cast<int>(profile.total_layers);
          if (layers_q8 < std::min(config.gpu_layers, max_offloadable)) {
            int layers_q4 = calculate_optimal_gpu_layers(profile, free_vram, config, KVCacheType::Q4_0);
            if (layers_q4 > layers_q8) {
              resolved_kv_cache_type = KVCacheType::Q4_0;
            }
          }
        }

        int optimal_layers = calculate_optimal_gpu_layers(profile, free_vram, config, resolved_kv_cache_type);
        if (optimal_layers < config.gpu_layers) {
          config.gpu_layers = optimal_layers;
          mp.n_gpu_layers = optimal_layers;
        }
      }
    }

    offloaded_layers = config.gpu_layers;
    if (total_layers > 0) offloaded_layers = std::min<int>(offloaded_layers, total_layers);
    if (config.gpu_layers <= 0) offloaded_layers = 0;

    struct ProgCtx { ProgressCallback* cb; } prog_ctx{&progress_cb};
    mp.progress_callback_user_data = &prog_ctx;
    mp.progress_callback = [](float p, void* u) -> bool {
      auto* c = static_cast<ProgCtx*>(u);
      return (c->cb && *(c->cb)) ? (*(c->cb))(p) : true;
    };

    model.reset(llama_model_load_from_file(config.model_path.c_str(), mp));
    if (!model) {
      std::fprintf(stderr, "%s%sError: Failed to load model from %s%s\n", TermColor::CLR_LINE, TermColor::RED, config.model_path.c_str(), TermColor::NORMAL);
      throw std::runtime_error("Unable to load model data from: " + config.model_path);
    }
    vocab = llama_model_get_vocab(model.get());
    rope_type = llama_model_rope_type(model.get());
    n_swa = llama_model_n_swa(model.get());

    enum llama_ftype ft = llama_model_ftype(model.get());
    const char* ft_name = llama_ftype_name(ft);
    if (ft_name && std::strlen(ft_name) > 0 && ft != LLAMA_FTYPE_GUESSED) {
      quantization_type = ft_name;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = config.context_size;
    if (safe_mode && config.ubatch_size > 32) {
      config.ubatch_size = 32;
    }
    config.ubatch_size = std::min(config.ubatch_size, config.batch_size);
    cp.n_batch = std::max(config.batch_size, config.ubatch_size);
    cp.n_ubatch = config.ubatch_size;
    cp.flash_attn_type = config.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (config.gpu_layers == 0 || !config.offload_kqv) cp.offload_kqv = false;

    cp.type_k = cp.type_v = (resolved_kv_cache_type == KVCacheType::Q8_0) ? GGML_TYPE_Q8_0 :
                            ((resolved_kv_cache_type == KVCacheType::Q4_0) ? GGML_TYPE_Q4_0 : GGML_TYPE_F16);

    uint32_t threads = config.threads;
    if (threads == 0) {
      uint32_t hw = std::thread::hardware_concurrency();
      threads = (hw > 2) ? (hw / 2) : std::max(1u, hw);
    }
    cp.n_threads = static_cast<int32_t>(threads);
    cp.n_threads_batch = static_cast<int32_t>(config.threads_batch ? config.threads_batch : threads);

    ctx.reset(llama_init_from_model(model.get(), cp));
    if (!ctx) throw std::runtime_error("Failed to initialize llama_context from model.");

    if (config.disable_thinking) {
      llama_token t1, t2;
      int n1 = llama_tokenize(vocab, "<think>", 7, &t1, 1, false, true);
      int n2 = llama_tokenize(vocab, "</think>", 8, &t2, 1, false, true);
      supports_disabling_thinking = (n1 == 1 && n2 == 1 &&
          (llama_vocab_get_attr(vocab, t1) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) &&
          (llama_vocab_get_attr(vocab, t2) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)));
      if (!supports_disabling_thinking) config.disable_thinking = false;
    }

    batch.init(config.batch_size, 0, 1);
    scratch_tokens_.reserve(config.context_size + 32);

    if (!config.mmproj_path.empty()) {
      load_multimodal_projector(config.mmproj_path);
    }
  }

  bool load_multimodal_projector(const std::string& path) {
    if (path.empty() || !model) return false;
    ctx_mtmd.reset();

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = (config.gpu_layers > 0);
    if (selected_devices_[0]) {
      mparams.device = selected_devices_[0];
    }
    mparams.n_threads = static_cast<int>(config.threads);
    mparams.flash_attn_type = config.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mparams.warmup = false;
    mparams.print_timings = config.verbose_logging;

    ctx_mtmd.reset(mtmd_init_from_file(path.c_str(), model.get(), mparams));
    if (!ctx_mtmd) {
      std::fprintf(stderr, "%sError: Failed to load multimodal projector from %s%s\n",
                   TermColor::RED, path.c_str(), TermColor::NORMAL);
      return false;
    }
    config.mmproj_path = path;
    std::fprintf(stderr, "%s[Multimodal projector loaded: %s (Vision: %s, Audio: %s)]%s\n",
                 TermColor::GREEN, path.c_str(),
                 mtmd_support_vision(ctx_mtmd.get()) ? "yes" : "no",
                 mtmd_support_audio(ctx_mtmd.get()) ? "yes" : "no",
                 TermColor::NORMAL);
    return true;
  }

  bool has_multimodal() const {
    return ctx_mtmd != nullptr;
  }

  bool supports_vision() const {
    return ctx_mtmd && mtmd_support_vision(ctx_mtmd.get());
  }

  bool supports_audio() const {
    return ctx_mtmd && mtmd_support_audio(ctx_mtmd.get());
  }

  std::string get_multimodal_marker() const {
    if (ctx_mtmd) {
      const char* m = mtmd_get_marker(ctx_mtmd.get());
      if (m && std::strlen(m) > 0) return std::string(m);
    }
    return mtmd_default_marker();
  }

  void configure_sampler(const InferenceParams& params) {
    smpl.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (params.repeat_penalty != 1.0f || params.presence_penalty != 0.0f) {
      llama_sampler_chain_add(smpl.get(), llama_sampler_init_penalties(
          llama_vocab_n_tokens(vocab), params.repeat_last_n, params.repeat_penalty, 0.0f,
          params.presence_penalty));
    }
    if (params.top_k > 0) llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(params.top_k));
    if (params.top_p < 1.0f) llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(params.top_p, 1));
    if (params.min_p > 0.0f) llama_sampler_chain_add(smpl.get(), llama_sampler_init_min_p(params.min_p, 1));
    if (params.temperature > 0.0f) {
      llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(params.temperature));
      llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.seed));
    } else {
      llama_sampler_chain_add(smpl.get(), llama_sampler_init_greedy());
    }
  }

  bool can_seq_add() const {
    return (rope_type != LLAMA_ROPE_TYPE_MROPE &&
            rope_type != LLAMA_ROPE_TYPE_IMROPE &&
            n_swa == 0);
  }

  // Synchronize, wipe the whole KV cache and synchronize again.
  void clear_kv_cache() {
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_synchronize(ctx.get());
    llama_memory_clear(mem, true);
    llama_synchronize(ctx.get());
  }

  void shift_context_if_needed(uint32_t additional_tokens) {
    if (n_past + additional_tokens < config.context_size) return;
    uint32_t n_keep = (config.context_size * 5) / 100;
    uint32_t n_latest = (config.context_size * 60) / 100;
    int32_t n_discard = std::max(static_cast<int32_t>(n_past - n_latest - n_keep),
                                 static_cast<int32_t>(n_past + additional_tokens - config.context_size));
    n_discard = std::min(n_discard, static_cast<int32_t>(n_past - n_keep));

    if (n_discard > 0) {
      llama_memory_t mem = llama_get_memory(ctx.get());
      if (can_seq_add() && llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard)) {
        llama_memory_seq_add(mem, 0, n_keep + n_discard, n_past, -n_discard);
        n_past -= n_discard;
      } else {
        clear_kv_cache();
        n_past = 0;
      }
    }

    if (n_past + additional_tokens >= config.context_size) {
      clear_kv_cache();
      n_past = 0;
    }
  }

  // Make room for `n` new tokens: shift the context when the model supports it,
  // otherwise reset the KV cache if the context would overflow.
  void prepare_context(uint32_t n) {
    if (can_seq_add()) {
      shift_context_if_needed(n);
    } else if (n_past + n > config.context_size) {
      clear_kv_cache();
      n_past = 0;
    }
    if (n_past + n > config.context_size) {
      clear_kv_cache();
      n_past = 0;
    }
  }

  // Prompt-evaluation chunk size: the configured batch size, capped at 128
  // tokens when offloading to the GPU (keeps individual submissions small).
  int32_t max_batch_size() const {
    int32_t max_batch = static_cast<int32_t>(config.batch_size);
    if (safe_mode && config.gpu_layers > 0 && max_batch > 128) max_batch = 128;
    return max_batch;
  }

  int tokenize(const std::string& text, bool add_special) const {
    if (!vocab || text.empty()) return 0;
    scratch_tokens_.resize(text.length() + 16);
    int n = llama_tokenize(vocab, text.c_str(), text.length(),
                           scratch_tokens_.data(), scratch_tokens_.size(), add_special, true);
    if (n < 0) {
      scratch_tokens_.resize(-n);
      n = llama_tokenize(vocab, text.c_str(), text.length(),
                         scratch_tokens_.data(), scratch_tokens_.size(), add_special, true);
    }
    return n;
  }
};

LLMEngine::LLMEngine(const EngineConfig& config, ProgressCallback progress_cb)
    : pimpl(std::make_unique<Impl>(config, progress_cb)) {}

LLMEngine::~LLMEngine() {
  if (!pimpl) return;
  // llama_free() synchronizes the Vulkan backend. If the device was lost, it
  // can throw from a noexcept unique_ptr destructor path and abort the process.
  llama_context* ctx = pimpl->ctx.release();
  if (ctx) {
    try {
      llama_free(ctx);
    } catch (...) {
      // Device is already gone; there is nothing left to synchronize.
    }
  }
  pimpl.reset();
}
LLMEngine::LLMEngine(LLMEngine&&) noexcept = default;
LLMEngine& LLMEngine::operator=(LLMEngine&&) noexcept = default;

const char* LLMEngine::GetChatTemplate() const {
  return (pimpl && pimpl->model) ? llama_model_chat_template(pimpl->model.get(), nullptr) : nullptr;
}

std::string LLMEngine::ApplyChatTemplate(const std::vector<ChatMessage>& history, bool add_generation_prompt) const {
  const char* tmpl = GetChatTemplate();
  if (tmpl && std::strlen(tmpl) > 0) {
    std::vector<llama_chat_message> chat;
    chat.reserve(history.size());
    for (const auto& msg : history) chat.push_back({msg.role.c_str(), msg.content.c_str()});
    std::vector<char> buf(4096);
    int res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_generation_prompt, buf.data(), buf.size());
    if (res > static_cast<int>(buf.size())) {
      buf.resize(res + 1);
      res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_generation_prompt, buf.data(), buf.size());
    }
    if (res > 0) return std::string(buf.data(), res);
  }

  std::string s_tmpl = tmpl ? std::string(tmpl) : "";
  struct Format { const char *turn_pfx, *role_mid, *turn_sfx, *ass_pfx; bool remap_model; };
  Format f = {"<|im_start|>", "\n", "<|im_end|>\n", "<|im_start|>assistant\n", false};

  if (s_tmpl.find("<|start_header_id|>") != std::string::npos) {
    f = {"<|start_header_id|>", "<|end_header_id|>\n\n", "<|eot_id|>", "<|start_header_id|>assistant<|end_header_id|>\n\n", false};
  } else if (s_tmpl.find("<start_of_turn>") != std::string::npos) {
    f = {"<start_of_turn>", "\n", "<end_of_turn>\n", "<start_of_turn>model\n", true};
  } else if (s_tmpl.find("<|turn>") != std::string::npos || s_tmpl.find("<turn|>") != std::string::npos) {
    f = {"<|turn>", "\n", "<turn|>\n", "<|turn>model\n", true};
  }

  std::string out = (s_tmpl.find("<|start_header_id|>") != std::string::npos) ? "<|begin_of_text|>" : "";
  std::string system_acc;
  for (const auto& m : history) {
    if (f.remap_model && m.role == "system") {
      if (!system_acc.empty()) system_acc += "\n\n";
      system_acc += m.content;
      continue;
    }
    std::string role = (f.remap_model && m.role == "assistant") ? "model" : m.role;
    std::string content = m.content;
    if (!system_acc.empty() && role == "user") {
      content = system_acc + "\n\n" + content;
      system_acc.clear();
    }
    out += f.turn_pfx + role + f.role_mid + content + f.turn_sfx;
  }
  if (!system_acc.empty()) {
    out += std::string(f.turn_pfx) + "user" + f.role_mid + system_acc + f.turn_sfx;
  }
  if (add_generation_prompt) out += f.ass_pfx;
  return out;
}

size_t LLMEngine::PruneChatHistory(std::vector<ChatMessage>& history, size_t target_tokens) const {
  size_t start_idx = (!history.empty() && history[0].role == "system") ? 1 : 0;
  size_t removed = 0;
  while (history.size() > start_idx + 2) {
    if (CountTokens(ApplyChatTemplate(history, true)) <= target_tokens) break;
    history.erase(history.begin() + start_idx, history.begin() + start_idx + 2);
    removed += 2;
  }
  return removed;
}

std::string LLMEngine::GetModelDescription() const {
  if (!pimpl || !pimpl->model) return "Unknown";
  char buf[256];
  int n = llama_model_desc(pimpl->model.get(), buf, sizeof(buf));
  return (n > 0) ? std::string(buf, n) : "Unknown";
}

std::string LLMEngine::GetQuantizationType() const {
  return pimpl ? pimpl->quantization_type : "Unknown";
}

std::string LLMEngine::GetKVCacheTypeShortName() const {
  return pimpl ? get_kv_cache_info(pimpl->resolved_kv_cache_type).short_name : "Unknown";
}

int LLMEngine::GetOffloadedLayers() const {
  return pimpl ? pimpl->offloaded_layers : 0;
}

uint32_t LLMEngine::GetTotalLayers() const {
  return pimpl ? pimpl->total_layers : 0;
}

void LLMEngine::ResetContext() {
  if (!pimpl || !pimpl->ctx) return;
  pimpl->clear_kv_cache();
  pimpl->n_past = 0;
  pimpl->abort_requested.store(false);
}

uint32_t LLMEngine::GetContextUsage() const { return pimpl ? pimpl->n_past : 0; }
uint32_t LLMEngine::GetContextSize() const { return pimpl ? pimpl->config.context_size : 0; }

void LLMEngine::AbortGeneration() {
  if (pimpl) pimpl->abort_requested.store(true);
}

size_t LLMEngine::CountTokens(const std::string& text) const {
  if (!pimpl) return 0;
  int n = pimpl->tokenize(text, false);
  return (n > 0) ? static_cast<size_t>(n) : 0;
}

bool LLMEngine::ProcessPrompt(const std::string& prompt, GenerationStats* stats) {
  return ProcessPrompt(prompt, {}, stats);
}

bool LLMEngine::ProcessPrompt(const std::string& prompt,
                              const std::vector<std::string>& media_paths,
                              GenerationStats* stats) {
  if (!pimpl || !pimpl->ctx) return false;
  pimpl->abort_requested.store(false);
  auto p_start = std::chrono::steady_clock::now();

  // Shared by the text-only and multimodal paths.
  auto fill_prompt_stats = [&](size_t n_tokens) {
    std::chrono::duration<double> p_dur = std::chrono::steady_clock::now() - p_start;
    if (stats) {
      stats->prompt_tokens = static_cast<int>(n_tokens);
      stats->prompt_eval_time_sec = p_dur.count();
      stats->prompt_tps = (p_dur.count() > 0.0) ? (n_tokens / p_dur.count()) : 0.0;
    }
  };

  std::string formatted_prompt = prompt;
  if (pimpl->config.disable_thinking && pimpl->supports_disabling_thinking) {
    formatted_prompt += "</think>\n";
  }

  const std::string marker = pimpl->get_multimodal_marker();
  bool prompt_has_marker = (pimpl->has_multimodal() && formatted_prompt.find(marker) != std::string::npos);

  if (media_paths.empty() && !prompt_has_marker) {
    bool add_special = (pimpl->n_past == 0);
    int n_tokens = pimpl->tokenize(formatted_prompt, add_special);
    if (n_tokens <= 0) return false;

    const int max_allowed = std::max(1, static_cast<int>(pimpl->config.context_size) - 32);
    if (n_tokens > max_allowed) {
      bool keep_bos = (add_special && n_tokens > 0);
      llama_token bos = pimpl->scratch_tokens_[0];
      size_t tail_count = keep_bos ? (max_allowed - 1) : max_allowed;
      std::memmove(pimpl->scratch_tokens_.data() + (keep_bos ? 1 : 0),
                   pimpl->scratch_tokens_.data() + (n_tokens - tail_count),
                   tail_count * sizeof(llama_token));
      if (keep_bos) pimpl->scratch_tokens_[0] = bos;
      n_tokens = max_allowed;
    }

    pimpl->prepare_context(n_tokens);

    int32_t max_batch = pimpl->max_batch_size();
    pimpl->batch.init(max_batch, 0, 1);
    const llama_token* tokens = pimpl->scratch_tokens_.data();

    for (int i = 0; i < n_tokens; ) {
      if (pimpl->abort_requested.load()) return false;
      int chunk = std::min(max_batch, n_tokens - i);

      pimpl->batch->n_tokens = chunk;
      std::copy_n(tokens + i, chunk, pimpl->batch->token);
      for (int j = 0; j < chunk; ++j) {
        pimpl->batch->pos[j] = pimpl->n_past + j;
        pimpl->batch->seq_id[j][0] = 0;
      }
      std::fill_n(pimpl->batch->n_seq_id, chunk, 1);
      std::fill_n(pimpl->batch->logits, chunk, false);
      if (i + chunk >= n_tokens) pimpl->batch->logits[chunk - 1] = true;

      if (llama_decode(pimpl->ctx.get(), *pimpl->batch) != 0) {
        std::fprintf(stderr, "\nError: llama_decode() failed in prompt evaluation\n");
        return false;
      }
      pimpl->n_past += chunk;
      i += chunk;
    }

    llama_synchronize(pimpl->ctx.get());
    fill_prompt_stats(n_tokens);
    return true;
  }

  // Multimodal evaluation path
  if (!pimpl->has_multimodal()) {
    std::fprintf(stderr, "%sError: No multimodal projector loaded. Use --mmproj <path> or /mmproj <path> to enable media input.%s\n",
                 TermColor::RED, TermColor::NORMAL);
    return false;
  }

  struct MediaCleanup {
    std::vector<mtmd_bitmap*> bitmaps;
    std::vector<mtmd_helper_video*> videos;
    ~MediaCleanup() {
      for (auto* b : bitmaps) if (b) mtmd_bitmap_free(b);
      for (auto* v : videos) if (v) mtmd_helper_video_free(v);
    }
  } media_cleanup;

  std::vector<const mtmd_bitmap*> bitmap_ptrs;
  mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();

  for (const auto& path : media_paths) {
    auto res = mtmd_helper_bitmap_init_from_file(pimpl->ctx_mtmd.get(), path.c_str(), false, init_opt);
    if (!res.bitmap) {
      std::fprintf(stderr, "%sError: Failed to load media from %s (unsupported format or unreadable file)%s\n",
                   TermColor::RED, path.c_str(), TermColor::NORMAL);
      return false;
    }
    media_cleanup.bitmaps.push_back(res.bitmap);
    if (res.video_ctx) media_cleanup.videos.push_back(res.video_ctx);
    bitmap_ptrs.push_back(res.bitmap);
  }

  size_t marker_count = 0;
  size_t pos = 0;
  while ((pos = formatted_prompt.find(marker, pos)) != std::string::npos) {
    ++marker_count;
    pos += marker.length();
  }
  if (marker_count < bitmap_ptrs.size()) {
    std::string prefix;
    for (size_t i = 0; i < (bitmap_ptrs.size() - marker_count); ++i) {
      prefix += marker + "\n";
    }
    formatted_prompt = prefix + formatted_prompt;
  }

  bool add_special = (pimpl->n_past == 0);
  mtmd_input_text text_in = {
    formatted_prompt.c_str(),
    formatted_prompt.length(),
    add_special,
    /* parse_special */ true
  };

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  int32_t tok_res = mtmd_tokenize(pimpl->ctx_mtmd.get(), chunks.ptr.get(), &text_in, bitmap_ptrs.data(), bitmap_ptrs.size());
  if (tok_res != 0) {
    std::fprintf(stderr, "%sError: Multimodal tokenization failed with code %d%s\n",
                 TermColor::RED, tok_res, TermColor::NORMAL);
    return false;
  }

  size_t n_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
  llama_pos n_pos = mtmd_helper_get_n_pos(chunks.ptr.get());
  if (n_tokens == 0) return false;

  pimpl->prepare_context(n_pos);

  int32_t max_batch = pimpl->max_batch_size();

  size_t n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
  for (size_t i = 0; i < n_chunks; ++i) {
    if (pimpl->abort_requested.load()) return false;
    bool logits_last = (i == n_chunks - 1);
    const auto* chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
    llama_pos next_n_past = pimpl->n_past;

    int32_t eval_res = mtmd_helper_eval_chunk_single(
        pimpl->ctx_mtmd.get(),
        pimpl->ctx.get(),
        chunk,
        pimpl->n_past,
        0,
        max_batch,
        logits_last,
        &next_n_past
    );
    if (eval_res != 0) {
      std::fprintf(stderr, "%sError: Failed evaluating multimodal chunk %zu (code: %d)%s\n",
                   TermColor::RED, i, eval_res, TermColor::NORMAL);
      return false;
    }
    pimpl->n_past = next_n_past;
  }

  llama_synchronize(pimpl->ctx.get());
  fill_prompt_stats(n_tokens);
  return true;
}

std::string LLMEngine::Generate(const InferenceParams& params,
                                TokenCallback token_cb,
                                GenerationStats* stats,
                                ThinkSpinnerCallback spinner_cb) {
  if (!pimpl || !pimpl->ctx || !pimpl->vocab) return "";
  pimpl->abort_requested.store(false);
  pimpl->configure_sampler(params);

  std::string response;
  response.reserve(512);
  int generated = 0;
  auto gen_start = std::chrono::steady_clock::now();

  ThinkingStreamFilter::Config f_cfg = params;
  if (pimpl->vocab) {
    for (llama_token tok : {llama_vocab_eos(pimpl->vocab), llama_vocab_eot(pimpl->vocab)}) {
      if (tok >= 0 && tok < llama_vocab_n_tokens(pimpl->vocab)) {
        char buf[128];
        int len = llama_token_to_piece(pimpl->vocab, tok, buf, sizeof(buf), 0, true);
        if (len > 0 && len < static_cast<int>(sizeof(buf))) {
          std::string piece(buf, len);
          if (!piece.empty() && std::find(f_cfg.stop_words.begin(), f_cfg.stop_words.end(), piece) == f_cfg.stop_words.end()) {
            f_cfg.stop_words.push_back(std::move(piece));
          }
        }
      }
    }
  }

  bool user_aborted = false;
  ThinkingStreamFilter filter(
      f_cfg,
      [&](std::string_view piece) {
        response.append(piece.data(), piece.size());
        if (token_cb && !token_cb(piece)) user_aborted = true;
      },
      spinner_cb,
      [&](std::string_view ui_piece) {
        if (token_cb && !token_cb(ui_piece)) user_aborted = true;
      });

  char stack_buf[128];
  std::vector<char> dyn_buf;

  while (true) {
    if (pimpl->abort_requested.load() || user_aborted ||
        (params.max_tokens > 0 && generated >= params.max_tokens)) {
      break;
    }

    float* logits = llama_get_logits_ith(pimpl->ctx.get(), -1);
    if (logits && (std::isnan(logits[0]) || std::isinf(logits[0]))) {
      std::fprintf(stderr, "\n%s[Warning: Numerical instability (NaN/Inf in model logits). Terminating generation.]%s\n",
                   TermColor::YELLOW, TermColor::NORMAL);
      break;
    }

    llama_token new_token_id = llama_sampler_sample(pimpl->smpl.get(), pimpl->ctx.get(), -1);
    llama_sampler_accept(pimpl->smpl.get(), new_token_id);
    if (llama_vocab_is_eog(pimpl->vocab, new_token_id)) break;

    int n = llama_token_to_piece(pimpl->vocab, new_token_id, stack_buf, sizeof(stack_buf), 0, true);
    std::string_view piece;
    if (n > 0) {
      piece = std::string_view(stack_buf, n);
    } else if (n < 0) {
      dyn_buf.resize(-n);
      int res = llama_token_to_piece(pimpl->vocab, new_token_id, dyn_buf.data(), dyn_buf.size(), 0, true);
      if (res > 0) piece = std::string_view(dyn_buf.data(), res);
    }

    generated++;
    if (!filter.ProcessPiece(piece) || user_aborted) break;

    if (pimpl->n_past + 1 >= pimpl->config.context_size && !pimpl->can_seq_add()) break;

    pimpl->shift_context_if_needed(1);
    pimpl->batch->n_tokens = 1;
    pimpl->batch->token[0] = new_token_id;
    pimpl->batch->pos[0] = pimpl->n_past;
    pimpl->batch->n_seq_id[0] = 1;
    pimpl->batch->seq_id[0][0] = 0;
    pimpl->batch->logits[0] = true;

    if (llama_decode(pimpl->ctx.get(), *pimpl->batch) != 0) {
      std::fprintf(stderr, "\nError: llama_decode() failed during generation\n");
      break;
    }
    pimpl->n_past += 1;
  }

  filter.Flush();

  std::chrono::duration<double> gen_dur = std::chrono::steady_clock::now() - gen_start;
  if (stats) {
    stats->generated_tokens = generated;
    stats->thinking_tokens = filter.GetThinkingTokens();
    stats->thinking_time_sec = filter.GetThinkingDurationSec();
    stats->generation_time_sec = gen_dur.count();
    stats->generation_tps = (gen_dur.count() > 0.0) ? (generated / gen_dur.count()) : 0.0;
  }
  return response;
}

std::vector<DeviceInfo> LLMEngine::GetAvailableDevices() {
  LogManager::instance().ensure_silent();
  ensure_backend_init();
  std::vector<DeviceInfo> devices;
  size_t count = ggml_backend_dev_count();
  for (size_t i = 0; i < count; ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    if (!dev) continue;
    DeviceInfo info;
    info.index = i;
    const char* n = ggml_backend_dev_name(dev);
    const char* d = ggml_backend_dev_description(dev);
    info.name = n ? n : "Unknown";
    info.description = d ? d : "";
    auto t = ggml_backend_dev_type(dev);
    info.type = (t == GGML_BACKEND_DEVICE_TYPE_CPU) ? "CPU" :
                (t == GGML_BACKEND_DEVICE_TYPE_GPU) ? "GPU" :
                (t == GGML_BACKEND_DEVICE_TYPE_IGPU) ? "iGPU" :
                (t == GGML_BACKEND_DEVICE_TYPE_ACCEL) ? "ACCEL" : "OTHER";
    ggml_backend_dev_memory(dev, &info.free_memory_bytes, &info.total_memory_bytes);
    devices.push_back(info);
  }
  return devices;
}

bool LLMEngine::LoadMultimodalProjector(const std::string& mmproj_path) {
  return pimpl ? pimpl->load_multimodal_projector(mmproj_path) : false;
}

bool LLMEngine::HasMultimodal() const {
  return pimpl ? pimpl->has_multimodal() : false;
}

bool LLMEngine::SupportsVision() const {
  return pimpl ? pimpl->supports_vision() : false;
}

bool LLMEngine::SupportsAudio() const {
  return pimpl ? pimpl->supports_audio() : false;
}

