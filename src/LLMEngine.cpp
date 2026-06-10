#include <LLMEngine.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>


// Constructor definition
LLMEngine::LLMEngine(const char *model_filepath, unsigned int context_window){
  // 1. Initialize the backend
  llama_backend_init();
  llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

  // 2. Load the model parameters and model (Updated API)
  llama_model_params model_params = llama_model_default_params();
  model = llama_model_load_from_file(model_filepath, model_params);
  if (!model) {
    std::cerr << "Error: Failed to load model from " << model_filepath << "\n";
    throw std::runtime_error("\nERROR: Unable to load model data!\n");
  }

  // 3. Extract the Vocabulary (New API concept)
  vocab = llama_model_get_vocab(model);

  // 4. Create the context (Updated API)
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = context_window;
  ctx = llama_init_from_model(model, ctx_params);

  n_past = 0;
  // llama_token eot_id = llama_vocab_eot(vocab);


}

// Destructor definition
LLMEngine::~LLMEngine(){
  // 9. Free resources (Updated model free)
  llama_sampler_free(smpl);
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
}


void LLMEngine::ProcessPrompt(std::string prompt){
  // 5. Tokenize the prompt (Now requires vocab, not model)
  std::vector<llama_token> tokens_list(prompt.length() + 4);
  int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);

  if (n_tokens < 0) {
    tokens_list.resize(-n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens_list.data(), tokens_list.size(), true, true);
  }
  tokens_list.resize(n_tokens);

  // 6. Initialize and populate the batch manually (Macros removed in latest master)
  batch = llama_batch_init(512, 0, 1);
  batch.n_tokens = tokens_list.size();

  for (size_t i = 0; i < tokens_list.size(); ++i) {
    batch.token[i] = tokens_list[i];
    batch.pos[i] = n_past + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = false;
  }

  // Calculate logits for the very last token in the prompt
  batch.logits[batch.n_tokens - 1] = true;

  // 7. Set up a simple greedy sampler (Updated API)
  smpl = llama_sampler_init_greedy();

  // std::cout << prompt;
}


// 8. Evaluation and Generation Loop
void LLMEngine::GenerationLoop(unsigned int max_context){
  // Loop until we run out of allocated context space
  while (n_past < max_context) {
    // Decode the current batch
    if (llama_decode(ctx, batch) != 0) {
      std::cerr << "\nError: llama_decode() failed (Context limit likely reached)\n";
      break;
    }

    n_past += batch.n_tokens;

    // Sample the next token
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, batch.n_tokens - 1);
    llama_sampler_accept(smpl, new_token_id);

    // Break naturally if the model outputs an End Of Generation (EOG) token
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      // std::cout<<"\nBreaking due to EOG tag"<<std::endl;
      break;
    }

    // Convert token to text and print it
    char buf[128];
    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n > 0) {
      std::cout << std::string(buf, n);
      std::cout.flush();
    }

    // Clear batch and prepare it for the single newly generated token
    batch.n_tokens = 1;
    batch.token[0] = new_token_id;
    batch.pos[0] = n_past;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
  }

  std::cout << "\n";

}











