#include <LLMEngine.hpp>

#include <algorithm>
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <readline/history.h>
#include <readline/readline.h>
#include <sstream>
#include <string>
#include <vector>

// Vulkan queue/env policy is handled by LLMEngine: the graphics queue is always
// allowed, and submission serialization is enabled only by --safe-mode.

using namespace TermColor;

static std::atomic<bool> g_interrupted{false};
static std::atomic<LLMEngine*> g_active_engine{nullptr};

void sigint_handler(int) {
  g_interrupted.store(true, std::memory_order_relaxed);
  if (LLMEngine* eng = g_active_engine.load(std::memory_order_relaxed)) {
    eng->AbortGeneration();
  }
}

static std::string get_history_filepath() {
  return "./.llm_engine_history";
}

#define PROMPT_STRING "\001\033[1;34m\002> \001\033[0m\002"

static bool read_user_prompt(std::string& out_prompt) {
  while (true) {
    char* raw = readline(PROMPT_STRING);
    if (!raw) {
      if (g_interrupted.load()) {
        g_interrupted.store(false);
        std::cout << "\n";
        continue;
      }
      return false;
    }
    out_prompt = raw;
    free(raw);
    return true;
  }
}

static void print_devices() {
  auto devices = LLMEngine::GetAvailableDevices();
  std::cout << BLUE << "Compute devices (" << devices.size() << "):\n" << NORMAL;
  for (const auto& dev : devices) {
    std::cout << " [" << dev.index << "] " << CYAN << dev.name << NORMAL << " (" << dev.type << ")\n";
    if (!dev.description.empty()) std::cout << "     " << dev.description << "\n";
    if (dev.total_memory_bytes > 0) {
      std::cout << "     " << std::fixed << std::setprecision(1)
                << (dev.free_memory_bytes / 1048576.0) << " MB free / "
                << (dev.total_memory_bytes / 1048576.0) << " MB total\n";
    }
  }
  std::cout << "\n";
}

// Shared REPL command list, used by both --help and the /help command.
static void print_repl_commands() {
  std::cout << "  /help                      Show this command list\n"
            << "  /image <path> [prompt]     Attach media and optionally run a prompt\n"
            << "  /clear-media               Clear attached media\n"
            << "  /mmproj <path>             Load or switch multimodal projector\n"
            << "  /system <prompt>           Set system prompt (resets context)\n"
            << "  /reset                     Reset conversation and KV cache\n"
            << "  /save <file>               Save conversation to file\n"
            << "  /exit                      Exit application\n";
}

static void print_usage(const char* prog) {
  std::cout << BLUE << "LLM Inference Engine\n" << NORMAL
            << "Usage:\n  " << prog << " [options] <model_path>\n  " << prog << " --list-devices\n\n"
            << "Options:\n"
            << "  -m, --model <path>         Path to GGUF model file\n"
            << "  --mmproj <path>            Path to multimodal projector (GGUF)\n"
            << "  -i, --image <path>         Attach image to session/prompt\n"
            << "  --media <path>             Attach image/audio to session/prompt\n"
            << "  -p, --prompt <text>        Non-interactive one-shot prompt\n"
            << "  -s, --system <text>        System prompt\n"
            << "  -c, --ctx-size <N>         Context window in tokens (default: 4096)\n"
            << "  -b, --batch-size <N>       Logical prompt batch size (default: 512)\n"
            << "  -ub, --ubatch-size <N>     Prefill micro-batch (default: auto; 128, or 256 for >12 GiB models)\n"
            << "  -t, --threads <N>          CPU generation threads (default: 0 = physical cores)\n"
            << "  --threads-batch <N>        CPU prefill threads (default: same as threads)\n"
            << "  -ngl, --gpu-layers <N>     Layers to offload to GPU (default: 99, auto-clamped)\n"
            << "  --cpu                      Run on CPU only\n"
            << "  -d, --device <id/name>     Select compute device by index or name\n"
            << "  --list-devices             List compute devices and exit\n"
            << "  --no-flash-attn            Disable Flash Attention\n"
            << "  --kv-type <format>         KV cache: auto, f16, q8_0, q4_0 (default: auto)\n"
            << "  --no-mmap                  Load directly into RAM instead of mmap\n"
            << "  --mlock                    Lock model memory in RAM\n"
            << "  --no-host                  Bypass GPU pinned host buffers (UMA default)\n"
            << "  --watchdog-budget <sec>    Vulkan submission budget (default: 2.0, range: 1.0-9.5s)\n"
            << "  --safe-mode                Conservative path for very large models\n"
            << "  --force-performance        Disable automatic safe-mode\n\n"
            << "Sampling Parameters:\n"
            << "  --temp <float>             Temperature (default: 0.8)\n"
            << "  --top-k <int>              Top-K sampling (default: 40)\n"
            << "  --top-p <float>            Top-P sampling (default: 0.95)\n"
            << "  --min-p <float>            Min-P sampling (default: 0.05)\n"
            << "  --repeat-penalty <float>   Repetition penalty (default: 1.1)\n"
            << "  --seed <uint>              RNG seed (default: random)\n"
            << "  -n, --max-tokens <int>     Maximum tokens (-1 = unlimited)\n\n"
            << "Output:\n"
            << "  --hide-thinking            Hide <think> tokens, show reasoning time\n"
            << "  --disable-thinking         Suppress thinking tags when supported\n"
            << "  --show-metrics             Show prefill/decode tokens/sec\n"
            << "  --log-file <path>          Write engine diagnostics to file (default: none)\n"
            << "  -v, --verbose              Mirror engine diagnostics to stderr\n"
            << "  -h, --help                 Show this help\n\n"
            << "REPL Commands:\n";
  print_repl_commands();
}

static bool run_inference(LLMEngine& engine, const std::string& prompt,
                          const std::vector<std::string>& media_paths,
                          const InferenceParams& params,
                          bool show_metrics, GenerationStats& stats, std::string& out_response) {
  if (!engine.ProcessPrompt(prompt, media_paths, &stats)) {
    if (g_interrupted.load()) {
      std::cout << YELLOW << "\n[Prompt evaluation interrupted]\n" << NORMAL;
      g_interrupted.store(false);
    } else {
      std::cerr << RED << "\nError: Prompt processing failed.\n" << NORMAL;
    }
    return false;
  }

  std::cout << GREEN;
  auto spinner_cb = [](int tokens, double) {
    if (tokens % 4 == 0) {
      const char sp[] = {'-', '\\', '|', '/'};
      std::cout << CLR_LINE << CYAN << "[Thinking " << sp[(tokens / 4) % 4] << "]" << NORMAL << std::flush;
    }
  };

  TokenCallback cb = [&](std::string_view piece) {
    std::cout << piece << std::flush;
    return !g_interrupted.load();
  };

  out_response = engine.Generate(params, cb, &stats, spinner_cb);
  std::cout << NORMAL << "\n";

  if (g_interrupted.load()) {
    std::cout << YELLOW << "[Generation interrupted by user (Ctrl+C)]\n" << NORMAL;
    g_interrupted.store(false);
    return false;
  }

  // An empty visible response is only a failure when nothing was generated at
  // all (e.g. immediate end-of-generation). With --hide-thinking, a response
  // consisting solely of reasoning yields an empty string but generated_tokens
  // is greater than zero, so it must not be treated as an error.
  if (out_response.empty()) {
    if (stats.generated_tokens == 0) return false;
    std::cout << YELLOW << "[No visible output: all generated tokens were filtered (hidden reasoning)]\n" << NORMAL;
  }

  if (show_metrics && stats.generation_time_sec > 0.0) {
    std::cout << MAGENTA << "[Metrics] prefill " << std::fixed << std::setprecision(1)
              << stats.prompt_tps << " t/s (" << stats.prompt_eval_time_sec << "s) | decode "
              << stats.generation_tps << " t/s (" << stats.generated_tokens << " tok in "
              << stats.generation_time_sec << "s)";
    if (stats.thinking_tokens > 0) {
      std::cout << " | think " << stats.thinking_tokens << " tok (" << stats.thinking_time_sec << "s)";
    }
    std::cout << "\n" << NORMAL;
  }
  return true;
}

static bool set_sampling_param(InferenceParams& p, std::string key, const std::string& val) {
  if (val.empty()) return false;
  while (!key.empty() && key.front() == '-') key.erase(0, 1);
  try {
    if (key == "temp") p.temperature = std::stof(val);
    else if (key == "top-p") p.top_p = std::stof(val);
    else if (key == "top-k") p.top_k = std::stoi(val);
    else if (key == "min-p") p.min_p = std::stof(val);
    else if (key == "repeat-penalty") p.repeat_penalty = std::stof(val);
    else if (key == "seed") p.seed = std::stoul(val);
    else if (key == "max-tokens" || key == "n") p.max_tokens = std::stoi(val);
    else return false;
    return true;
  } catch (...) { return false; }
}

int main(int argc, char **argv) {
  std::signal(SIGINT, sigint_handler);

  EngineConfig config;
  InferenceParams params;
  bool show_metrics = false, list_devices_only = false;
  std::string one_shot_prompt, system_prompt;
  std::vector<std::string> staged_media;

  auto get_val = [&](int& i, const std::string& opt) -> std::string {
    if (i + 1 >= argc || (argv[i + 1][0] == '-' && std::strlen(argv[i + 1]) > 1 && !std::isdigit(argv[i + 1][1]))) {
      std::cerr << RED << "Error: Option '" << opt << "' requires a valid value.\n\n" << NORMAL;
      print_usage(argv[0]);
      std::exit(1);
    }
    return argv[++i];
  };

  // Parses an integer option and enforces a valid range. Negative values are
  // rejected for unsigned options (e.g. -b -1 used to wrap around to a huge
  // batch size and abort inside llama_decode).
  auto parse_num = [&](int& i, const std::string& opt, int min_val = INT_MIN, int max_val = INT_MAX) {
    std::string val = get_val(i, opt);
    try {
      int v = std::stoi(val);
      if (v < min_val || v > max_val) {
        std::cerr << RED << "Error: Option '" << opt << "' expects a value in the range ["
                  << min_val << ", " << max_val << "], got: '" << val << "'\n\n" << NORMAL;
        print_usage(argv[0]);
        std::exit(1);
      }
      return v;
    } catch (...) {
      std::cerr << RED << "Error: Option '" << opt << "' expects an integer, got: '" << val << "'\n\n" << NORMAL;
      print_usage(argv[0]);
      std::exit(1);
    }
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
    else if (arg == "--list-devices") list_devices_only = true;
    else if (arg == "-m" || arg == "--model") config.model_path = get_val(i, arg);
    else if (arg == "--mmproj") config.mmproj_path = get_val(i, arg);
    else if (arg == "-i" || arg == "--image" || arg == "--media") staged_media.push_back(get_val(i, arg));
    else if (arg == "-p" || arg == "--prompt") one_shot_prompt = get_val(i, arg);
    else if (arg == "-s" || arg == "--system") system_prompt = get_val(i, arg);
    else if (arg == "-c" || arg == "--ctx-size") config.context_size = parse_num(i, arg, 512, 1 << 24);
    else if (arg == "-b" || arg == "--batch-size") config.batch_size = parse_num(i, arg, 1, 65536);
    else if (arg == "-ub" || arg == "--ubatch-size") config.ubatch_size = parse_num(i, arg, 1, 65536);
    else if (arg == "-t" || arg == "--threads") config.threads = parse_num(i, arg, 0, 1024);
    else if (arg == "--threads-batch") config.threads_batch = parse_num(i, arg, 0, 1024);
    else if (arg == "-ngl" || arg == "--gpu-layers") config.gpu_layers = parse_num(i, arg);
    else if (arg == "--safe-mode") config.safe_mode = true;
    else if (arg == "--force-performance") config.force_performance = true;
    else if (arg == "--cpu") config.gpu_layers = 0;
    else if (arg == "-d" || arg == "--device") {
      std::string d = get_val(i, arg);
      try { config.device_index = std::stoi(d); } catch (...) { config.device_name = d; }
    } else if (arg == "--no-flash-attn") config.flash_attn = false;
    else if (arg == "--kv-type") {
      std::string kv = get_val(i, arg);
      if (kv == "q8_0") config.kv_cache_type = KVCacheType::Q8_0;
      else if (kv == "q4_0") config.kv_cache_type = KVCacheType::Q4_0;
      else if (kv == "f16") config.kv_cache_type = KVCacheType::F16;
      else if (kv == "auto") config.kv_cache_type = KVCacheType::AUTO;
      else {
        std::cerr << RED << "Unknown KV cache type: " << kv << " (valid: auto, f16, q8_0, q4_0)\n" << NORMAL;
        return 1;
      }
    } else if (arg == "--no-mmap") config.use_mmap = false;
    else if (arg == "--mlock") config.use_mlock = true;
    else if (arg == "--no-host") config.no_host = true;
    else if (arg == "--watchdog-budget") {
      std::string w = get_val(i, arg);
      try {
        config.watchdog_budget_sec = std::clamp(std::stod(w), 1.0, 9.5);
      } catch (...) {
        std::cerr << RED << "Error: Invalid watchdog budget value: '" << w << "'\n\n" << NORMAL;
        print_usage(argv[0]);
        return 1;
      }
    }
    else if (set_sampling_param(params, arg, (i + 1 < argc) ? argv[i + 1] : "")) { ++i; }
    else if (arg == "--hide-thinking") params.hide_thinking = true;
    else if (arg == "--disable-thinking") config.disable_thinking = true;
    else if (arg == "--show-metrics") show_metrics = true;
    else if (arg == "--log-file") config.log_file = get_val(i, arg);
    else if (arg == "-v" || arg == "--verbose") config.verbose_logging = true;
    else if (config.model_path.empty() && arg[0] != '-') config.model_path = arg;
    else {
      std::cerr << RED << "Unknown option: " << arg << NORMAL << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (list_devices_only) {
    print_devices();
    return 0;
  }

  if (config.model_path.empty() || config.context_size < 512) {
    std::cerr << RED << (config.model_path.empty() ? "Error: No model path specified.\n\n"
                                                   : "Error: Context size must be at least 512 tokens.\n") << NORMAL;
    print_usage(argv[0]);
    return 1;
  }

  try {
    auto progress_cb = [](float p) -> bool {
      static int last_pct = -1;
      int pct = static_cast<int>(p * 100.0f);
      if (pct == last_pct) return true;
      last_pct = pct;
      if (pct < 100 && pct % 10 != 0) return true;
      std::cerr << CLR_LINE << CYAN << "[Loading " << pct << "%]" << NORMAL << std::flush;
      if (pct == 100) std::cerr << "\n";
      return true;
    };
    LLMEngine engine(config, progress_cb);
    g_active_engine.store(&engine, std::memory_order_release);

    if (!one_shot_prompt.empty()) {
      std::vector<ChatMessage> history;
      if (!system_prompt.empty()) history.push_back({"system", system_prompt});
      history.push_back({"user", one_shot_prompt});
      std::string formatted = engine.ApplyChatTemplate(history, true);
      if (formatted.empty()) formatted = one_shot_prompt + "\n";
      GenerationStats stats;
      std::string response;
      bool ok = run_inference(engine, formatted, staged_media, params, show_metrics, stats, response);
      g_active_engine.store(nullptr, std::memory_order_release);
      return ok ? 0 : 1;
    }

    std::string hist_file = get_history_filepath();
    read_history(hist_file.c_str());

    const std::string model_desc = engine.GetModelDescription();
    const std::string model_quant = engine.GetQuantizationType();
    std::cout << CYAN << "Model: " << NORMAL << model_desc;
    if (model_desc.find(model_quant) == std::string::npos) {
      std::cout << CYAN << " | Quant: " << NORMAL << model_quant;
    }
    std::cout << CYAN << " | KV: " << NORMAL << engine.GetKVCacheTypeShortName()
              << CYAN << " | Ctx: " << NORMAL << engine.GetContextSize()
              << CYAN << " | Backend: " << NORMAL;
    if (engine.GetOffloadedLayers() > 0) {
      std::cout << "Vulkan (" << engine.GetOffloadedLayers() << "/" << engine.GetTotalLayers() << " layers)";
    } else {
      std::cout << "CPU";
    }
    std::cout << "\n";
    if (engine.HasMultimodal()) {
      std::cout << CYAN << "Multimodal: " << NORMAL
                << (engine.SupportsVision() ? "vision" : "")
                << (engine.SupportsVision() && engine.SupportsAudio() ? " + " : "")
                << (engine.SupportsAudio() ? "audio" : "") << "\n";
    }
    if (!staged_media.empty()) {
      std::cout << CYAN << "Media: " << NORMAL << staged_media.size() << " staged\n";
    }
    std::cout << "\n";

    std::vector<ChatMessage> history;
    if (!system_prompt.empty()) history.push_back({"system", system_prompt});

    std::string evaluated_text;

    rl_variable_bind("enable-bracketed-paste", "on");

    while (true) {
      g_interrupted.store(false);
      std::string input;
      if (!read_user_prompt(input)) { std::cout << "\n"; break; }
      if (input.empty()) continue;
      add_history(input.c_str());

      if (input[0] == '/') {
        std::istringstream iss(input);
        std::string cmd;
        iss >> cmd;
        // Set when /image carries an inline prompt: the prompt may itself start
        // with '/', in which case it must be sent to the model instead of being
        // treated as another slash command.
        bool inline_prompt_ready = false;

        auto reset_session = [&]() {
          engine.ResetContext();
          history.clear();
          if (!system_prompt.empty()) history.push_back({"system", system_prompt});
          evaluated_text.clear();
          staged_media.clear();
        };

        // List attached media for /image with no arguments.
        auto list_media = [&](const char* empty_msg) {
          if (staged_media.empty()) {
            std::cout << CYAN << empty_msg << NORMAL;
            return;
          }
          std::cout << CYAN << "\nMedia (" << staged_media.size() << "):\n" << NORMAL;
          for (size_t idx = 0; idx < staged_media.size(); ++idx) {
            std::cout << "  " << (idx + 1) << ". " << staged_media[idx] << "\n";
          }
          std::cout << "\n";
        };

        if (cmd == "/exit") break;
        if (cmd == "/help") {
          std::cout << "\n";
          print_repl_commands();
          std::cout << "\n";
        } else if (cmd == "/reset") {
          reset_session();
          std::cout << GREEN << "[Context reset]\n" << NORMAL;
        } else if (cmd == "/image") {
          std::string rest;
          std::getline(iss, rest);
          size_t first = rest.find_first_not_of(" \t");
          if (first == std::string::npos) {
            list_media("No media attached.\n");
            continue;
          }
          rest = rest.substr(first);
          std::string media_path, inline_prompt;
          if (rest.front() == '"' || rest.front() == '\'') {
            char quote = rest.front();
            size_t qend = rest.find(quote, 1);
            if (qend != std::string::npos) {
              media_path = rest.substr(1, qend - 1);
              inline_prompt = rest.substr(qend + 1);
            } else {
              media_path = rest.substr(1);
            }
          } else {
            size_t space_pos = rest.find_first_of(" \t");
            if (space_pos != std::string::npos) {
              media_path = rest.substr(0, space_pos);
              inline_prompt = rest.substr(space_pos + 1);
            } else {
              media_path = rest;
            }
          }
          size_t pfirst = inline_prompt.find_first_not_of(" \t");
          inline_prompt = (pfirst != std::string::npos) ? inline_prompt.substr(pfirst) : "";

          if (!std::filesystem::exists(media_path)) {
            std::cout << RED << "Error: media file not found: " << media_path << "\n" << NORMAL;
            continue;
          }

          if (!engine.HasMultimodal()) {
            std::cout << YELLOW << "[Warning: no multimodal projector loaded; use /mmproj <path>]\n" << NORMAL;
          }

          staged_media.push_back(media_path);
          std::cout << GREEN << "[Media: " << staged_media.size() << " staged]\n" << NORMAL;

          if (!inline_prompt.empty()) {
            input = inline_prompt;
            inline_prompt_ready = true;
          } else {
            continue;
          }
        } else if (cmd == "/clear-media") {
          staged_media.clear();
          std::cout << GREEN << "[Media cleared]\n" << NORMAL;
          continue;
        } else if (cmd == "/mmproj") {
          std::string mm_path;
          iss >> mm_path;
          if (mm_path.empty()) {
            std::cout << (engine.HasMultimodal() ? "Multimodal: loaded" : "Multimodal: not loaded") << "\n";
          } else if (!std::filesystem::exists(mm_path)) {
            std::cout << RED << "Error: file not found: " << mm_path << "\n" << NORMAL;
          } else if (engine.LoadMultimodalProjector(mm_path)) {
            config.mmproj_path = mm_path;
            std::cout << GREEN << "[Multimodal projector loaded]\n" << NORMAL;
          }
          continue;
        } else if (cmd == "/system") {
          std::string new_sys;
          std::getline(iss, new_sys);
          size_t first = new_sys.find_first_not_of(" \t");
          if (first != std::string::npos) {
            system_prompt = new_sys.substr(first);
            reset_session();
            std::cout << GREEN << "[System prompt updated]\n" << NORMAL;
          } else {
            std::cout << "System prompt: \"" << system_prompt << "\"\n";
          }
        } else if (cmd == "/save") {
          std::string save_path;
          if (iss >> save_path) {
            std::ofstream ofs(save_path);
            if (ofs.is_open()) {
              for (const auto& msg : history) ofs << "=== " << msg.role << " ===\n" << msg.content << "\n\n";
              std::cout << GREEN << "[Saved: " << save_path << "]\n" << NORMAL;
            } else {
              std::cout << RED << "Error: Could not open file for writing.\n" << NORMAL;
            }
          } else {
            std::cout << RED << "Usage: /save <filepath>\n" << NORMAL;
          }
        } else {
          std::cout << RED << "Unknown command: " << cmd << ". Type /help for command list.\n" << NORMAL;
        }
        if (input[0] == '/' && !inline_prompt_ready) continue;
      }

      if (engine.GetContextUsage() >= (engine.GetContextSize() * 75 / 100)) {
        size_t removed = engine.PruneChatHistory(history, engine.GetContextSize() * 35 / 100);
        if (removed > 0) {
          std::cout << YELLOW << "[Context pruned: " << (removed / 2) << " turns]\n" << NORMAL;
          engine.ResetContext();
          evaluated_text.clear();
        }
      }

      std::vector<std::string> current_media = std::move(staged_media);
      staged_media.clear();

      history.push_back({"user", input});
      std::string new_prompt = engine.ApplyChatTemplate(history, true);
      std::string diff = (!evaluated_text.empty() && new_prompt.rfind(evaluated_text, 0) == 0 && current_media.empty())
          ? new_prompt.substr(evaluated_text.length())
          : (engine.ResetContext(), evaluated_text.clear(), new_prompt);

      GenerationStats stats;
      std::string response;
      if (!run_inference(engine, diff, current_media, params, show_metrics, stats, response)) {
        history.pop_back();
        engine.ResetContext();
        evaluated_text.clear();
        std::cout << YELLOW << "[Context reset]\n" << NORMAL;
        continue;
      }

      history.push_back({"assistant", response});
      evaluated_text += diff + response;
    }

    g_active_engine.store(nullptr, std::memory_order_release);
    write_history(hist_file.c_str());
  } catch (const std::exception &e) {
    std::cerr << RED << "\nFatal Error: " << e.what() << NORMAL << "\n";
    return 1;
  }
  return 0;
}
