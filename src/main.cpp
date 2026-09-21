#include <LLMEngine.hpp>
#include <cstring>
#include <iostream>
#include <readline/history.h>
#include <readline/readline.h>
#include <vector>

#define DEFAULT_CONTEXT_LEN 4096

std::string apply_chat_template(const char *tmpl,
                                const std::vector<ChatMessage> &history,
                                bool add_ass) {
  std::vector<llama_chat_message> chat;
  for (const auto &msg : history) {
    chat.push_back({msg.role.c_str(), msg.content.c_str()});
  }
  int length = 4096;
  std::vector<char> buf(length);
  int res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass,
                                      buf.data(), length);
  if (res > length) {
    buf.resize(res);
    res = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass,
                                    buf.data(), res);
  }
  if (res < 0) {
    return "";
  }
  return std::string(buf.data(), res);
}

int main(int argc, char **argv) {
  bool disable_thinking = false;
  bool hide_thinking = false;
  int gpu_layers = 99;
  bool show_metrics = false;
  unsigned int context_size = DEFAULT_CONTEXT_LEN;
  std::string model_path = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--disable-thinking") {
      disable_thinking = true;
    } else if (arg == "--hide-thinking") {
      hide_thinking = true;
    } else if (arg == "--cpu-backend") {
      gpu_layers = 0;
    } else if (arg == "--gpu-layers" && i + 1 < argc) {
      gpu_layers = std::stoi(argv[++i]);
    } else if (arg == "--show-metrics") {
      show_metrics = true;
    } else if (arg == "--ctx-size" && i + 1 < argc) {
      context_size = std::stoi(argv[++i]);
    } else if (model_path.empty()) {
      model_path = arg;
    } else {
      std::cout << "Usage: " << argv[0]
                << " <model_path> [--disable-thinking] [--hide-thinking] [--cpu-backend] [--gpu-layers N] [--show-metrics] [--ctx-size N]\n";
      return 1;
    }
  }

  if (model_path.empty()) {
    std::cout << "Usage: " << argv[0]
              << " <model_path> [--disable-thinking] [--hide-thinking] [--cpu-backend] [--gpu-layers N] [--show-metrics] [--ctx-size N]\n"
              << std::endl;
    return 1;
  }

  try {
    LLMEngine llm(model_path.c_str(), context_size, disable_thinking, gpu_layers);

    const char *tmpl = llm.GetChatTemplate();
    if (tmpl == nullptr) {
      tmpl = ""; // Provide empty string to fallback or let
                 // llama_chat_apply_template handle it
    }

    std::vector<ChatMessage> history;
    double total_time = 0.0;
    int total_tokens = 0;

    while (1) {
      char *command;
      command = readline(BLUE "\n> " NORMAL);
      if (command == NULL) {
        clearerr(stdin);
        std::cout << std::endl;
        continue;
      }
      if (strlen(command) > 0)
        add_history(command);

      std::string prompt(command);
      if (prompt == "/exit") {
        free(command);
        break;
      }

      std::string prev_prompt = "";
      if (!history.empty()) {
        prev_prompt = apply_chat_template(tmpl, history, false);
      }

      history.push_back({"user", prompt});

      std::string new_prompt = apply_chat_template(tmpl, history, true);

      std::string diff = new_prompt;
      if (new_prompt.empty()) {
        // Fallback if template application fails
        diff = "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n";
      } else if (new_prompt.length() >= prev_prompt.length() &&
                 prev_prompt.length() > 0) {
        // The new prompt typically appends to the old one
        diff = new_prompt.substr(prev_prompt.length());
      }

      llm.ProcessPrompt(diff);
      std::cout << GREEN;
      int current_tokens = 0;
      double current_time = 0.0;
      std::string response = llm.GenerationLoop(context_size, hide_thinking, &current_tokens, &current_time);
      std::cout << NORMAL;
      
      if (show_metrics && current_time > 0.0) {
        double tps = current_tokens / current_time;
        std::cout << MAGENTA << "\n[Metrics] Speed: " << tps << " tokens/s (" << current_tokens << " tokens in " << current_time << "s)\n" << NORMAL;
      }
      
      total_tokens += current_tokens;
      total_time += current_time;

      history.push_back({"assistant", response});

      free(command);
    }
    
    if (show_metrics && total_time > 0.0) {
      double avg_tps = total_tokens / total_time;
      std::cout << MAGENTA << "\n[Session Metrics] Average Speed: " << avg_tps << " tokens/s (" << total_tokens << " tokens in " << total_time << "s)\n" << NORMAL;
    }
  } catch (const std::exception &e) {
    std::cerr << RED << "\n" << e.what() << NORMAL << "\n";
    return 1;
  }

  return 0;
}
