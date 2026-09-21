#include <LLMEngine.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <ctime>
#include <fcntl.h>
#include <unistd.h>

class StreamRedirector {
public:
  int console_fd = -1;
  StreamRedirector(const char *filename) {
    console_fd = dup(fileno(stdout));
    fflush(stdout);
    fflush(stderr);

    log_fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (log_fd != -1) {
      auto now = std::chrono::system_clock::now();
      auto time = std::chrono::system_clock::to_time_t(now);
      char time_str[64];
      strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S",
               std::localtime(&time));

      char buf[256];
      int len = snprintf(buf, sizeof(buf),
                         "\n\n=== LLM Engine Started at %s ===\n", time_str);
      if (len > 0) {
        if (write(log_fd, buf, len) < 0) {
        } // ignore write errors here
      }

      old_stderr = dup(fileno(stderr));
      dup2(log_fd, fileno(stderr));
      old_stdout = dup(fileno(stdout));
      dup2(log_fd, fileno(stdout));
    }
  }

  ~StreamRedirector() {
    if (log_fd != -1) {
      fflush(stdout);
      fflush(stderr);
      if (old_stdout != -1) {
        dup2(old_stdout, fileno(stdout));
        close(old_stdout);
      }
      if (old_stderr != -1) {
        dup2(old_stderr, fileno(stderr));
        close(old_stderr);
      }
      if (console_fd != -1) {
        close(console_fd);
      }
      close(log_fd);
    }
  }

private:
  int old_stdout = -1;
  int old_stderr = -1;
  int log_fd = -1;
};

// Constructor definition
LLMEngine::LLMEngine(const char *model_filepath, unsigned int context_window,
                     bool disable_thinking, int gpu_layers) {
  this->disable_thinking = disable_thinking;
  StreamRedirector redirector("llm_engine.log");

  std::atomic<bool> loading{true};
  auto load_start = std::chrono::steady_clock::now();

  std::thread spinner([&]() {
    const char spinner_chars[] = {'-', '\\', '|', '/'};
    int idx = 0;
    while (loading) {
      char buf[64];
      int len = snprintf(buf, sizeof(buf),
                         "\r\033[2K" CYAN "[Loading Model %c]" NORMAL,
                         spinner_chars[idx]);
      if (len > 0) {
        if (write(redirector.console_fd, buf, len) < 0) {
        }
      }
      idx = (idx + 1) % 4;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // Backend Initialization
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  // Loading model parameters and model
  llama_model_params model_params = llama_model_default_params();
  static ggml_backend_dev_t cpu_devices[2] = {nullptr, nullptr};

  if (gpu_layers > 0) {
    model_params.n_gpu_layers = gpu_layers;
  } else {
    model_params.n_gpu_layers = 0;
    cpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    model_params.devices = cpu_devices;
  }
  model = llama_model_load_from_file(model_filepath, model_params);
  if (!model) {
    loading = false;
    spinner.join();
    std::cerr << "Error: Failed to load model from " << model_filepath << "\n";
    throw std::runtime_error("\nERROR: Unable to load model data!\n");
  }

  // Vocabulary Extraction
  vocab = llama_model_get_vocab(model);

  // Context creation
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = context_window;
  if (gpu_layers == 0) {
      ctx_params.offload_kqv = false;
  }
  ctx = llama_init_from_model(model, ctx_params);

  loading = false;
  spinner.join();

  auto load_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> load_duration = load_end - load_start;
  char buf[128];
  int len = snprintf(buf, sizeof(buf),
                     "\r\033[2K" CYAN "[Model loaded in %.1fs]\n" NORMAL,
                     load_duration.count());
  if (len > 0) {
    if (write(redirector.console_fd, buf, len) < 0) {
    }
  }

  if (this->disable_thinking) {
    std::vector<llama_token> test_tokens1(4);
    int n1 = llama_tokenize(vocab, "<think>", 7, test_tokens1.data(),
                            test_tokens1.size(), false, true);

    std::vector<llama_token> test_tokens2(4);
    int n2 = llama_tokenize(vocab, "</think>", 8, test_tokens2.data(),
                            test_tokens2.size(), false, true);

    if (n1 == 1 && n2 == 1) {
      llama_token tok1 = test_tokens1[0];
      llama_token tok2 = test_tokens2[0];

      enum llama_token_attr attr1 = llama_vocab_get_attr(vocab, tok1);
      enum llama_token_attr attr2 = llama_vocab_get_attr(vocab, tok2);

      const char *text1 = llama_vocab_get_text(vocab, tok1);
      const char *text2 = llama_vocab_get_text(vocab, tok2);

      bool is_special1 =
          (attr1 & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED));
      bool is_special2 =
          (attr2 & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED));

      bool text_matches = (text1 && std::string(text1) == "<think>") &&
                          (text2 && std::string(text2) == "</think>");

      if (is_special1 && is_special2 && text_matches) {
        this->supports_disabling_thinking = true;
      } else {
        const char *msg =
            "\r\033[2K\033[1;31mWarning: Model does not support turning off "
            "thinking steps. Disregarding flag.\033[0m\n";
        if (write(redirector.console_fd, msg, strlen(msg)) < 0) {
        }
        this->disable_thinking = false;
        this->supports_disabling_thinking = false;
      }
    } else {
      const char *msg =
          "\r\033[2K\033[1;31mWarning: Model does not support turning off "
          "thinking steps. Disregarding flag.\033[0m\n";
      if (write(redirector.console_fd, msg, strlen(msg)) < 0) {
      }
      this->disable_thinking = false;
      this->supports_disabling_thinking = false;
    }
  }

  n_past = 0;
  // llama_token eot_id = llama_vocab_eot(vocab);
}

LLMEngine::~LLMEngine() {
  llama_sampler_free(smpl);
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}

const char *LLMEngine::GetChatTemplate() const {
  return llama_model_chat_template(model, nullptr);
}

void LLMEngine::ProcessPrompt(std::string prompt) {
  if (this->disable_thinking && this->supports_disabling_thinking) {
    prompt += "</think>\n";
  }

  bool add_special = (n_past == 0);
  std::vector<llama_token> tokens_list(prompt.length() + 4);
  int n_tokens =
      llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(),
                     tokens_list.size(), add_special, true);

  if (n_tokens < 0) {
    tokens_list.resize(-n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                              tokens_list.data(), tokens_list.size(),
                              add_special, true);
  }
  tokens_list.resize(n_tokens);

  // initialize and populate batches
  if (smpl != nullptr) {
    llama_sampler_free(smpl);
    smpl = nullptr;
  }
  if (batch.token != nullptr) {
    llama_batch_free(batch);
  }
  batch = llama_batch_init(tokens_list.size(), 0, 1);
  batch.n_tokens = tokens_list.size();

  for (size_t i = 0; i < tokens_list.size(); ++i) {
    batch.token[i] = tokens_list[i];
    batch.pos[i] = n_past + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
  }

  // logits calculation for last token in prompt
  batch.logits[batch.n_tokens - 1] = true;

  // use simple greedy sampler if one isn't already set
  if (smpl == nullptr) {
    smpl = llama_sampler_init_greedy();
  }
  // std::cout << prompt;
}

// Generation Loop
std::string LLMEngine::GenerationLoop(unsigned int max_context,
                                      bool hide_thinking,
                                      int* out_tokens,
                                      double* out_time) {
  std::string response = "";
  std::string print_buffer = "";
  bool in_thinking = false;
  int think_tokens = 0;
  int generated = 0;
  auto gen_start = std::chrono::steady_clock::now();
  auto think_start = gen_start;

  // Loop until we run out of allocated context space
  while (n_past < max_context) {
    // Decode the current batch
    if (llama_decode(ctx, batch) != 0) {
      std::cerr
          << "\nError: llama_decode() failed (Context limit likely reached)\n";
      break;
    }

    n_past += batch.n_tokens;

    // Sample the next token
    llama_token new_token_id =
        llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
    llama_sampler_accept(smpl, new_token_id);
    generated++;

    // Break naturally if the model outputs an End Of Generation (EOG) token
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      break;
    }

    // Convert token to text and print
    char buf[128];
    int n =
        llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n > 0) {
      std::string piece(buf, n);
      response += piece;
      print_buffer += piece;

      while (!print_buffer.empty()) {
        if (!in_thinking) {
          size_t pos = print_buffer.find("<think>");
          if (pos != std::string::npos) {
            std::string before = print_buffer.substr(0, pos);
            std::cout << before;
            if (!hide_thinking) {
              std::cout << CYAN << "<think>";
            } else {
              // Just move to a new line for the spinner
              std::cout << "\n" << std::flush;
            }
            in_thinking = true;
            think_start = std::chrono::steady_clock::now();
            think_tokens = 0;
            print_buffer = print_buffer.substr(pos + 7);
          } else {
            size_t partial_pos = std::string::npos;
            const char *tag = "<think>";
            for (size_t i = 1; i <= 7; ++i) {
              if (print_buffer.length() >= i &&
                  print_buffer.substr(print_buffer.length() - i) ==
                      std::string(tag, i)) {
                partial_pos = print_buffer.length() - i;
                break;
              }
            }
            if (partial_pos != std::string::npos) {
              std::string safe_to_print = print_buffer.substr(0, partial_pos);
              std::cout << safe_to_print;
              print_buffer = print_buffer.substr(partial_pos);
              break;
            } else {
              std::cout << print_buffer;
              print_buffer = "";
            }
          }
        } else {
          size_t pos = print_buffer.find("</think>");
          if (pos != std::string::npos) {
            std::string before = print_buffer.substr(0, pos);

            auto think_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> think_duration =
                think_end - think_start;
            char time_buf[32];
            snprintf(time_buf, sizeof(time_buf), "%.1f",
                     think_duration.count());

            if (!hide_thinking) {
              std::cout << before << "</think>\n"
                        << CYAN << "[Thought for " << time_buf << "s]\n"
                        << GREEN;
            } else {
              // Clear the spinner line completely and show thought time
              std::cout << "\r\033[2K" << CYAN << "[Thought for " << time_buf
                        << "s]\n"
                        << GREEN << std::flush;
            }
            in_thinking = false;
            print_buffer = print_buffer.substr(pos + 8);
          } else {
            size_t partial_pos = std::string::npos;
            const char *tag = "</think>";
            for (size_t i = 1; i <= 8; ++i) {
              if (print_buffer.length() >= i &&
                  print_buffer.substr(print_buffer.length() - i) ==
                      std::string(tag, i)) {
                partial_pos = print_buffer.length() - i;
                break;
              }
            }
            if (partial_pos != std::string::npos) {
              std::string safe_to_print = print_buffer.substr(0, partial_pos);
              if (!hide_thinking)
                std::cout << safe_to_print;
              print_buffer = print_buffer.substr(partial_pos);
              break;
            } else {
              if (!hide_thinking)
                std::cout << print_buffer;
              print_buffer = "";
            }
          }
        }
      }

      if (in_thinking && hide_thinking) {
        think_tokens++;
        if (think_tokens % 4 == 0) {
          const char spinner[] = {'-', '\\', '|', '/'};
          int spinner_idx = (think_tokens / 4) % 4;
          std::cout << "\r\033[2K" << CYAN << "[Thinking "
                    << spinner[spinner_idx] << "]" << std::flush;
        }
      }

      std::cout.flush();
    }

    // Clear batch and prepare for the single newly generated token
    batch.n_tokens = 1;
    batch.token[0] = new_token_id;
    batch.pos[0] = n_past;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
  }

  if (out_tokens) *out_tokens = generated;
  if (out_time) {
    auto gen_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> gen_duration = gen_end - gen_start;
    *out_time = gen_duration.count();
  }

  if (!print_buffer.empty()) {
    if (!in_thinking || !hide_thinking) {
      std::cout << print_buffer;
    }
  }

  std::cout << "\n";
  return response;
}
