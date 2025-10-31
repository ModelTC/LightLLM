#pragma once

#include <filesystem>
#include <string_view>

auto inline constexpr default_index_directory_name = "i";
auto inline constexpr default_block_directory_name = "b";
auto inline constexpr default_block_name_split_position = 4;

auto inline to_block_path(std::filesystem::path path, //
                          std::string_view name,      //
                          std::size_t position = default_block_name_split_position) -> std::filesystem::path {
  path /= name.substr(0, position);
  if (position < name.size()) {
    path /= name.substr(position);
  }
  return path;
}
