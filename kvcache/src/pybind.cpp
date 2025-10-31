#include "config.h"
#include "core/error.h"
#include "core/task.h"
#include "service/local_cache_service.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(MODULE_NAME, m) {
  using cache::service::LocalCacheService;
  using cache::storage::LocalStorageEngine;
  using cache::task::CacheTask;
  using cache::task::State;

  py::enum_<State>(m, "State")
      .value("Initial", State::Initial)
      .value("Working", State::Working)
      .value("Finished", State::Finished)
      .value("Aborted", State::Aborted)
      .export_values();

  py::class_<LocalStorageEngine::HashInfo, std::shared_ptr<LocalStorageEngine::HashInfo>>(m, "LocalCacheHashInfo");

  py::class_<CacheTask, std::shared_ptr<CacheTask>>(m, "Task")
      .def("ready", &CacheTask::ready, "Check if task is ready")
      .def("state", &CacheTask::state, "Get task block states");

  py::class_<LocalCacheService>(m, "LocalCacheService")
      .def(py::init<const std::string &, std::size_t, std::size_t, const torch::Tensor &, std::size_t>(),
           py::arg("file"),
           py::arg("storage_size"),
           py::arg("num_of_shard"),
           py::arg("kvcache"),
           py::arg("num_workers"))
      .def("run", &LocalCacheService::run)
      .def("query", &LocalCacheService::query)
      .def("az5", &LocalCacheService::az5)
      .def("create", &LocalCacheService::create)
      .def("get_hash_info", &LocalCacheService::get_hash_info)
      .def("set_hash_info", &LocalCacheService::set_hash_info)
      .def("block_size", &LocalCacheService::block_size)
      .def("page_size", &LocalCacheService::page_size);
}