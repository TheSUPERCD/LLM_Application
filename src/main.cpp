#include <LLMEngine.hpp>
#include <iostream>
#include<readline/readline.h>
#include<readline/history.h>

#define GREEN "\033[1;32m"
#define BLUE "\033[1;34m"
#define NORMAL "\033[0m"
#define RED "\033[1;31m"
#define CYAN "\033[1;36m"
#define MAGENTA "\033[1;35m"

#define MAX_CONTEXT_LEN 128000

int main(int argc, char ** argv) {
  if(argc < 2){
    std::cout<<"Usage: "<<argv[0]<<" <model_path>\n"<<std::endl;
    return 1;
  }
  LLMEngine gemma4(argv[1], MAX_CONTEXT_LEN);
  std::string cmdtemplate_begin = "<|turn>user\n";
  std::string cmdtemplate_end   = "<turn|>\n<|turn>model\n";
  while(1){
    char *command;
    command = readline(BLUE "\ngemma-shell> " NORMAL);
    // if uses enter Ctrl+D, we don't exit -- since the project said "only exit when Ctrl+C is pressed"
    if(command == NULL)
      continue;
    // add the history functionality to the shell (the arrow keys to bring old commands into view) -- very useful!!
    if(strlen(command)>0)
      add_history(command);

    std::string prompt(command);
    if(prompt == "/exit")
      break;
    gemma4.ProcessPrompt(cmdtemplate_begin + prompt + cmdtemplate_end);
    std::cout<<GREEN;
    gemma4.GenerationLoop(MAX_CONTEXT_LEN);
    std::cout<<NORMAL;
  }
  return 0;
}
