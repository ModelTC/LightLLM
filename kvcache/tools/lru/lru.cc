#include "common.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// A key-only LRU cache.
class lru_cache_t {
  std::size_t capacity;         // Maximum number of items the cache can hold
  std::list<std::string> order; // List to maintain access order (most recent at front)
  std::unordered_map<std::string_view, std::list<std::string>::iterator> table; // Hash table for O(1) access

public:
  // Constructor that initializes the cache with given capacity
  explicit lru_cache_t(std::size_t capacity) : capacity(capacity) {
    table.reserve(capacity); // Pre-allocate memory for the hash table
  }

  // Adds or updates an item in the cache
  // Returns true if an item was evicted, false otherwise
  // Optional evicted parameter can store the evicted value
  auto touch(std::string value, std::string *evicted = nullptr) -> bool {
    // Reject empty strings
    if (value.empty()) {
      return false;
    }

    if (auto iter = table.find(value); iter != table.end()) { // If item already exists in cache
      order.splice(order.begin(), order, iter->second);
      return false;
    }

    auto evict = capacity <= order.size();
    if (not evict) {
      order.push_front(std::move(value));

    } else {
      table.erase(order.back());

      if (evicted != nullptr) { // If evicted pointer provided, store the evicted value
        *evicted = std::move(order.back());
      }

      order.splice(order.begin(), order, std::prev(order.end()));
      order.front() = std::move(value);
    }

    // New value added.
    table.emplace(order.front(), order.begin());
    return evict;
  };
};

// Class for reading continuously growing files (e.g., log files)
// Features: Tracks read position, efficiently handles file appends
class continuous_line_reader_t {
  bool end_of_line = false;    // Flag indicating if a complete line was read
  std::streamoff position = 0; // Current read position (file offset)
  std::ifstream file;          // File input stream (could be upgraded to mmap)
  std::string buffer;          // Temporary buffer for raw data reading
  std::string line;            // Current line being constructed (may span multiple reads)

public:
  enum { default_line_capacity = 512 }; // Default line buffer capacity

  // Constructor: Pre-allocates line buffer capacity
  explicit continuous_line_reader_t(std::size_t line_capacity = default_line_capacity) {
    buffer.reserve(line_capacity);
    line.reserve(line_capacity);
  }

  // Path constructor: Opens file after initialization
  template <typename S>
  explicit continuous_line_reader_t(S const &path, std::size_t line_capacity = default_line_capacity)
      : continuous_line_reader_t(line_capacity) {
    open(path);
  }

  // Opens a file, closes any previously opened file
  template <typename S> auto open(S const &path) -> bool {
    if (file.is_open()) {
      file.close();
    }
    file.open(path, std::ios::in | std::ios::binary); // Binary mode to avoid text conversion
    position = 0;                                     // Reset read position
    return file.is_open();
  }

  // Closes the file
  auto close() -> void { file.close(); }

  // Core method: Gets the next line
  auto getline(std::string_view &output) -> bool {
    if (not file.is_open()) {
      return false;
    }

    // If previous call got a complete line, reset state for new line
    if (end_of_line) {
      end_of_line = false;
      line.clear();
    }

    // Seek to last read position then read a line (may be incomplete if EOF has no newline)
    file.seekg(position);
    if (not std::getline(file, buffer)) {
      file.clear(); // Clear possible error states (e.g. EOF)
      return false;
    }

    // Successfully got new data position
    if (auto g = file.tellg(); g > 0) {
      position = g;
      end_of_line = true;
      line += buffer;
      output = line;
      return true;
    }

    // Handle EOF case (no newline at end)
    position += static_cast<std::streamoff>(buffer.size());
    line += buffer;
    return false;
  }
};

// Thread-safe task executor class that processes tasks in parallel
template <typename T> class executor_t {
  bool running = true;
  std::deque<T> tasks;
  std::vector<std::thread> threads;
  std::condition_variable condition;
  std::mutex mutex;

  // Worker thread function that processes tasks
  auto execute() -> void {
    for (auto keep = true; keep;) {
      auto task = std::optional<T>(); // Holds current task
      {
        // Lock and wait for tasks or shutdown signal
        auto lock = std::unique_lock(mutex);
        condition.wait(lock, [&] { return not tasks.empty() or not running; });

        if (tasks.empty()) {
          keep = running; // Exit when no tasks and not running
        } else {
          task.emplace(std::move(tasks.front()));
          tasks.pop_front();
        }
      }

      if (task) {
        (*task)(); // Execute the task
      }
    }
  }

public:
  // Delete copy/move constructors and assignment operators
  executor_t(executor_t &&) = delete;
  executor_t(executor_t const &) = delete;
  executor_t &operator=(executor_t &&) = delete;
  executor_t &operator=(executor_t const &) = delete;

  // Constructor creates worker threads
  explicit executor_t(std::size_t size) {
    size = std::max<std::size_t>(1, size); // Ensure at least 1 thread
    threads.reserve(size);
    while (size-- > 0) {
      threads.emplace_back(&executor_t::execute, this);
    }
  }

  // Destructor stops threads and cleans up
  ~executor_t() {
    {
      std::scoped_lock _(mutex);
      running = false; // Signal threads to stop
    }
    condition.notify_all();
    for (auto &t : threads) {
      t.join();
    }
  }

  // Add new task to the queue
  template <typename... U> auto push(U &&...args) -> bool {
    {
      std::scoped_lock _(mutex);
      if (not running) [[unlikely]] {
        return false;
      }
      tasks.emplace_back(std::forward<U>(args)...);
    }
    condition.notify_one();
    return true;
  }
};

class value_queue_t /* mpsc */ {
  std::vector<std::string> queue;
  std::mutex mutex;

public:
  auto push(std::string value) -> void {
    std::scoped_lock _(mutex);
    queue.emplace_back(std::move(value));
  }

  auto take() -> std::vector<std::string> {
    auto output = std::vector<std::string>();
    {
      std::scoped_lock _(mutex);
      output.reserve(queue.capacity());
      std::swap(queue, output);
    }
    return output;
  }
};

// Sort by string length then alphabetical order
struct by_order_then_alphanumeric_t {
  auto operator()(std::filesystem::path const &lhs, std::filesystem::path const &rhs) const noexcept {
    auto &l = lhs.native();
    auto &r = rhs.native();
    return l.size() != r.size() ? l.size() < r.size() : l < r;
  }
};

// Class for reading files in rolling fashion (like log files)
class rolling_reader_t {
  using path_t = std::filesystem::path;                                  // Path type alias
  using time_t = std::filesystem::file_time_type;                        // File time type alias
  using list_t = std::map<path_t, time_t, by_order_then_alphanumeric_t>; // Sorted file list

  list_t queue;                    // Collection of files with their timestamps
  list_t::iterator cursor;         // Current position in the file queue
  std::chrono::seconds timeout;    // Time threshold for file expiration
  continuous_line_reader_t reader; // Helper for reading file lines

private:
  // Move cursor to next valid readable file
  auto move_cursor() -> bool {
    if (cursor == queue.end()) {
      return false;
    }
    // Iterate through remaining files to find next readable one
    for (auto next = std::next(cursor); next != queue.end(); ++next) {
      if (cursor = next; reader.open(cursor->first)) {
        return true;
      }
    }
    return false;
  }

public:
  // Constructor initializes with initial file path, time and timeout
  explicit rolling_reader_t(path_t const &path, time_t time, std::chrono::seconds timeout)
      : queue{{path, time}},   //
        cursor(queue.begin()), //
        reader(path),          //
        timeout(timeout) {}

  // Read next line from current or next available file
  auto getline(std::string_view &out) -> bool {
    while (not reader.getline(out)) {
      auto minimum = time_t::clock::now() - timeout;    // Calculate expiration time
      if (cursor->second < minimum and move_cursor()) { // Check if file expired
        continue;
      }
      return false; // No more lines available
    }
    return true;
  }

  // Remove a file from tracking
  auto remove(path_t const &key) -> bool {
    if (cursor != queue.end() and cursor->first == key) { // If removing current file
      cursor = queue.erase(cursor);
      reader.close();
      // Find next readable file
      while (cursor != queue.end() and not reader.open(cursor->first)) {
        ++cursor;
      }
      return true;
    }
    return queue.erase(key) > 0; // Remove non-current file
  }

  // Update or add a file to tracking
  auto update(path_t const &key, time_t time) -> bool {
    auto constexpr less = by_order_then_alphanumeric_t();
    if (not queue.empty() and less(key, queue.begin()->first)) {
      return false; // Reject files older than our earliest tracked file
    }
    queue[key] = time;           // Insert or update file
    if (cursor == queue.end()) { // Reset cursor if needed
      cursor = queue.begin();
    }
    return true;
  }

  // Check if no files are being tracked
  auto empty() -> bool { return queue.empty(); }
};

// Reads a certain number of strings from input and forwards them to output.
struct index_task_t {
  // NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
  value_queue_t &output;
  rolling_reader_t &input;
  std::atomic_int &reference;
  // NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

  ~index_task_t() { reference.fetch_sub(1, std::memory_order_relaxed); }
  explicit index_task_t(value_queue_t &output, rolling_reader_t &input, std::atomic_int &reference) noexcept
      : output(output), input(input), reference(reference) {
    reference.fetch_add(1, std::memory_order_relaxed);
  }

  auto operator()() -> void {
    auto line = std::string_view();
    for (auto count = 0; count < 1024 and input.getline(line); ++count) {
      output.push(std::string(line));
    }
  }
};

} // namespace

// Print
template <typename... T> auto echo(T &&...args) -> void {
  // Avoiding disorder when printing from multiple threads
  std::cout << ((std::ostringstream() << ... << std::forward<T>(args)) << "\n").str();
}

auto clean(                             //
    std::filesystem::path const &block, //
    lru_cache_t &cache,                 //
    value_queue_t &queue,               //
    std::atomic_bool &running,          //
    std::atomic_uint &count             //
    ) -> void                           //
{
  while (running.load(std::memory_order_relaxed)) {
    for (auto &&input : queue.take()) {
      if (auto output = std::string(); cache.touch(input, &output)) {
        if (auto code = std::error_code(); std::filesystem::remove(to_block_path(block, output), code)) {
          count.fetch_add(1, std::memory_order_relaxed);
          // echo("EVICT: ", output);
        } else {
          echo("EVICT: ", output, ", but: ", code.message());
        }
      }
    }
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1ms);
  }
}

struct options_t {
  std::filesystem::path index = default_index_directory_name;       // Directory to index files.
  std::filesystem::path block = default_block_directory_name;       // Directory to data block files.
  std::chrono::minutes expired_deadline = std::chrono::minutes(30); // Ignoring outdated index files.
  std::chrono::seconds updated_deadline = std::chrono::seconds(6);  // Delay in rolling to next file.
  std::size_t cache_capacity = 128ULL * 1024 * 1024;                // Maximum number of data block files.
  std::size_t worker_number = 16;                                   // Number of threads reading index.
};

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto loop(options_t const &o) -> void {
  using namespace std::chrono_literals;
  namespace fs = std::filesystem;
  using time_t = fs::file_time_type;
  using void_t = std::shared_ptr<void>;

  auto scanned = std::map<fs::path, time_t, by_order_then_alphanumeric_t>();
  auto tracked = std::map<fs::path, time_t, by_order_then_alphanumeric_t>();
  auto pool = std::map<fs::path, rolling_reader_t>();
  auto queue = value_queue_t();
  auto cache = lru_cache_t(o.cache_capacity);
  auto executor = executor_t<index_task_t>(o.worker_number);
  auto running = std::atomic_bool(true);
  auto removed = std::atomic_uint(0);
  auto remover = std::thread(clean, o.block, std::ref(cache), std::ref(queue), std::ref(running), std::ref(removed));
  auto latch = std::atomic_int(0);
  auto defer = void_t(nullptr, [&](...) noexcept(false) {
    running.store(false, std::memory_order_relaxed);
    remover.join();
    echo("END");
  });

  while (running.load(std::memory_order_relaxed)) {
    // [1] scan index files.
    auto deadline = time_t::clock::now() - o.expired_deadline;
    for (auto &entry : fs::directory_iterator(o.index)) {
      // may switch to inotify
      if (entry.path().has_extension() and //
          entry.path().has_stem() and      //
          entry.is_regular_file())         //
      {
        if (auto mtime = entry.last_write_time(); deadline < mtime) {
          scanned[entry.path()] = mtime;
          continue;
        }
      }
      if (auto code = std::error_code(); std::filesystem::remove(entry, code)) {
        echo("REMOVE file: ", entry);
      } else {
        echo("REMOVE file: ", entry, ", but: ", code.message());
      }
    }

    // [2] wait executor to finish all tasks.
    while (latch.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(1ms);
    }

    // [3] update task info.
    for (auto &[path, _] : tracked) {
      if (scanned.find(path) == scanned.end()) {
        // delete task
        if (auto iter = pool.find(path.stem()); iter != pool.end()) {
          if (auto &task = iter->second; task.remove(path) and task.empty()) {
            echo("TASK remove: ", iter->first);
            pool.erase(iter);
          }
        }
      }
    }
    for (auto &[path, time] : scanned) {
      if (auto [iter, created] = pool.try_emplace(path.stem(), path, time, o.updated_deadline); not created) {
        iter->second.update(path, time);
      } else {
        echo("TASK create: ", iter->first);
      }
    }
    tracked.clear();
    std::swap(scanned, tracked);

    // [4] start tasks from pool.
    for (auto &[key, value] : pool) {
      executor.push(queue, value, latch);
    }

    // [5] wait next round
    echo("EVICTED: ", removed.load(std::memory_order_relaxed));
    std::this_thread::sleep_for(1s);
  }
}

int main(int argc, char *argv[]) {
  auto o = options_t{};
  for (auto i = 1; i + 1 < argc; i += 2) {
    auto key = std::string_view(argv[i]);
    /*  */ if (key == "--index") {
      o.index = argv[i + 1];
    } else if (key == "--block") {
      o.block = argv[i + 1];
    } else if (key == "--capacity") {
      o.cache_capacity = std::atoi(argv[i + 1]);
    }
  }

  if (o.index.empty() or o.block.empty() or o.cache_capacity <= 0) {
    std::cerr << "Usage:\n"
                 "\t--index index-dir-path\n"
                 "\t--block block-dir-path\n"
                 "\t--capacity lru-capacity\n";
    return 1;
  }

  std::cout << "OPTIONS:"           //
            << " index=" << o.index //
            << " block=" << o.block //
            << " capacity=" << o.cache_capacity << "\n";

  std::filesystem::create_directories(o.index);
  std::filesystem::create_directories(o.block);
  loop(o);
  return 0;
}
