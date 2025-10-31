#include "common.h"

#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <mutex>
#include <random>

namespace {

class index_writer_t {
  auto constexpr static flag = std::ios::out | std::ios::binary;

  std::size_t count = 0;
  std::size_t index = 0;
  std::filesystem::path base;
  std::ofstream output;
  std::mutex mutex;

public:
  explicit index_writer_t(std::filesystem::path dir) : base(std::move(dir)) {
    std::filesystem::create_directories(base.parent_path());
    output.open(base.replace_extension(std::to_string(index)), flag);
  }

  auto write(std::string const &value) -> void {
    auto static constexpr page = 16U; // limit to 2^16 = 65536

    std::scoped_lock _(mutex);
    if (count++ >> page != index) {
      index = (count - 1) >> page;
      output.close();
      output.open(base.replace_extension(std::to_string(index)), flag);
      // what if fail to open
    }
    output << value << '\n';
  }
};

class block_loader_t {
  auto constexpr static flag = std::ios::out | std::ios::binary;
  std::filesystem::path base;

public:
  explicit block_loader_t(std::filesystem::path dir) : base(std::move(dir)) {
    std::filesystem::create_directories(base);
  }

  auto write(std::string const &name, char const *data, std::size_t size) -> bool {
    auto path = to_block_path(base, name);
    auto ofs = std::ofstream(path, flag);
    if (not ofs.is_open() and std::filesystem::create_directory(path.parent_path())) {
      ofs.open(path, flag);
    }
    return static_cast<bool>(ofs.write(data, static_cast<std::streamsize>(size)));
  }

  auto read(std::string const &name, char *data, std::size_t size) -> bool {
    auto path = to_block_path(base, name);
    auto ifs = std::ifstream(path, std::ios::in | std::ios::binary);
    return static_cast<bool>(ifs.read(data, static_cast<std::streamsize>(size)));
  }

  auto path() const noexcept -> std::filesystem::path const & { return base; }
};

class block_access_t { // StorageEngine
  index_writer_t index;
  block_loader_t block;

public:
  explicit block_access_t(std::filesystem::path const &base, std::string const &host_name)
      : index(base / default_index_directory_name / host_name), //
        block(base / default_block_directory_name) {}

  auto query(std::string const &name) -> bool {
    auto path = to_block_path(block.path(), name);
    return std::filesystem::exists(path);
  }

  auto write(std::string const &name, char const *data, std::size_t size) -> bool {
    index.write(name);
    return block.write(name, data, size);
  }

  auto read(std::string const &name, char *data, std::size_t size) -> bool {
    index.write(name);
    return block.read(name, data, size);
  }
};

} // namespace

auto main() -> int {
  auto make = [s = std::string(64, '\0'),                //
               r = std::mt19937(std::random_device{}()), //
               d = std::uniform_int_distribution<std::size_t>(0, 16)]() mutable -> std::string const & {
    for (auto &c : s) {
      c = "0123456789abcdef"[d(r) % 16];
    }
    return s;
  };
  auto block = block_access_t(".", "host-" + make().substr(0, 27));
  for (auto i = 0U; i != ~0U; ++i) {
    auto const &text = make();
    block.write(text, text.data(), text.size());
  }
  return 0;
}
