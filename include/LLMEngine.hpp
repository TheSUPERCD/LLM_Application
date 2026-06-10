#pragma once
#ifndef LLM_ENGINE_HPP
#define LLM_ENGINE_HPP


#include <llama.h>
#include <string>


class LLMEngine{
public:
  LLMEngine(const char *model_filepath, unsigned int context_window);
  ~LLMEngine();

  void ProcessPrompt(std::string prompt);
  void GenerationLoop(unsigned int max_context);

private:
  llama_model *model;
  const llama_vocab *vocab;
  llama_context *ctx;
  llama_sampler *smpl;
  llama_batch batch;
  unsigned long int n_past;

};


#endif
