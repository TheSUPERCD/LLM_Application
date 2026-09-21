#pragma once
#ifndef LLM_ENGINE_HPP
#define LLM_ENGINE_HPP

#include <llama.h>
#include <string>
#include <vector>

#define GREEN "\033[1;32m"
#define BLUE "\033[1;34m"
#define NORMAL "\033[0m"
#define RED "\033[1;31m"
#define CYAN "\033[1;36m"
#define MAGENTA "\033[1;35m"

struct ChatMessage {
  std::string role;
  std::string content;
};

class LLMEngine {
public:
  LLMEngine(const char *model_filepath, unsigned int context_window,
            bool disable_thinking = false, int gpu_layers = 99);
  ~LLMEngine();

  void ProcessPrompt(std::string prompt);
  std::string GenerationLoop(unsigned int max_context,
                             bool hide_thinking = false,
                             int* out_tokens = nullptr,
                             double* out_time = nullptr);
  const char *GetChatTemplate() const;

private:
  llama_model *model;
  const llama_vocab *vocab;
  llama_context *ctx;
  llama_sampler *smpl = nullptr;
  llama_batch batch = {};
  unsigned long int n_past;
  bool disable_thinking = false;
  bool supports_disabling_thinking = false;
};

#endif
