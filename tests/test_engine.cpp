#include <LLMEngine.hpp>

#include <iostream>
#include <string>
#include <string_view>

namespace {

int failures = 0;

const std::string kThinkStart = "<think" ">";
const std::string kThinkEnd = "</think" ">";

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    std::cerr << "FAIL line " << line << ": " << expression << "\n";
    ++failures;
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

void test_plain_text() {
  std::string out;
  ThinkingStreamFilter filter(
      FilterConfig{},
      [&](std::string_view piece) { out.append(piece); });
  CHECK(filter.ProcessPiece("hello world"));
  filter.Flush();
  CHECK(out == "hello world");
}

void test_visible_thinking_split_across_pieces() {
  std::string out;
  std::string ui;
  ThinkingStreamFilter filter(
      FilterConfig{},
      [&](std::string_view piece) { out.append(piece); },
      nullptr,
      [&](std::string_view piece) { ui.append(piece); });

  CHECK(filter.ProcessPiece("A<thi"));
  CHECK(filter.ProcessPiece("nk>secret</thi"));
  CHECK(filter.ProcessPiece("nk>B"));
  filter.Flush();

  const std::string expected = std::string("A") + kThinkStart + "secret" + kThinkEnd + "B";
  CHECK(out == expected);
  CHECK(ui.find("Thought for") != std::string::npos);
}

void test_hidden_thinking_split_across_pieces() {
  std::string out;
  std::string ui;
  FilterConfig config;
  config.hide_thinking = true;
  ThinkingStreamFilter filter(
      config,
      [&](std::string_view piece) { out.append(piece); },
      nullptr,
      [&](std::string_view piece) { ui.append(piece); });

  CHECK(filter.ProcessPiece("A<thi"));
  CHECK(filter.ProcessPiece("nk>secret</thi"));
  CHECK(filter.ProcessPiece("nk>B"));
  filter.Flush();

  CHECK(out == "AB");
  CHECK(ui.find("Thought for") != std::string::npos);
}

void test_stop_word_split_across_pieces() {
  std::string out;
  ThinkingStreamFilter filter(
      FilterConfig{},
      [&](std::string_view piece) { out.append(piece); });

  CHECK(filter.ProcessPiece("hello<|im_"));
  CHECK(!filter.ProcessPiece("end|>world"));
  filter.Flush();

  CHECK(out == "hello");
}

void test_stop_word_inside_thinking() {
  std::string out;
  std::string ui;
  FilterConfig config;
  config.hide_thinking = true;
  ThinkingStreamFilter filter(
      config,
      [&](std::string_view piece) { out.append(piece); },
      nullptr,
      [&](std::string_view piece) { ui.append(piece); });

  const std::string start = std::string("<think") + ">abc";
  CHECK(filter.ProcessPiece(start));
  CHECK(!filter.ProcessPiece("<|im_end|>tail"));
  filter.Flush();
  CHECK(out.empty());
  CHECK(ui.empty());
}

void test_unterminated_hidden_thinking_flushes_banner() {
  std::string out;
  std::string ui;
  FilterConfig config;
  config.hide_thinking = true;
  ThinkingStreamFilter filter(
      config,
      [&](std::string_view piece) { out.append(piece); },
      nullptr,
      [&](std::string_view piece) { ui.append(piece); });

  const std::string input = std::string("<think") + ">abc";
  CHECK(filter.ProcessPiece(input));
  filter.Flush();

  CHECK(out.empty());
  CHECK(ui.find("Thought for") != std::string::npos);
}

void test_banner_falls_back_to_emitter() {
  std::string out;
  FilterConfig config;
  config.hide_thinking = true;
  ThinkingStreamFilter filter(
      config,
      [&](std::string_view piece) { out.append(piece); });

  const std::string input = std::string("<think") + ">abc";
  CHECK(filter.ProcessPiece(input));
  filter.Flush();

  CHECK(out.find("Thought for") != std::string::npos);
}

void test_custom_tags() {
  std::string out;
  std::string ui;
  FilterConfig config;
  config.think_start_tag = "<reason>";
  config.think_end_tag = "</reason>";
  ThinkingStreamFilter filter(
      config,
      [&](std::string_view piece) { out.append(piece); },
      nullptr,
      [&](std::string_view piece) { ui.append(piece); });

  CHECK(filter.ProcessPiece("pre<reason>why</reason>post"));
  filter.Flush();
  CHECK(out == "pre<reason>why</reason>post");
  CHECK(ui.find("Thought for") != std::string::npos);
}

}  // namespace

int main() {
  test_plain_text();
  test_visible_thinking_split_across_pieces();
  test_hidden_thinking_split_across_pieces();
  test_stop_word_split_across_pieces();
  test_stop_word_inside_thinking();
  test_unterminated_hidden_thinking_flushes_banner();
  test_banner_falls_back_to_emitter();
  test_custom_tags();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all ThinkingStreamFilter tests passed\n";
  return 0;
}
