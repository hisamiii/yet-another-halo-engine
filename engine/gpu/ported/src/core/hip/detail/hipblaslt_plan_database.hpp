#ifndef GUFO_CORE_HIP_DETAIL_HIPBLASLT_PLAN_DATABASE_HPP_
#define GUFO_CORE_HIP_DETAIL_HIPBLASLT_PLAN_DATABASE_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gufo::hip::detail {

inline constexpr std::uint32_t kHipblasLtPlanDatabaseSchemaVersion = 1;

enum class HipblasLtPlanDataType : std::uint8_t {
  kBfloat16 = 1,
};

struct HipblasLtPlanDatabaseKey {
  std::string hardware_fingerprint;
  std::uint32_t hip_runtime_version{0};
  std::uint32_t hipblaslt_version{0};

  [[nodiscard]] bool operator==(const HipblasLtPlanDatabaseKey&) const =
      default;
};

struct HipblasLtPlanRecord {
  std::uint64_t batch_size{0};
  std::uint64_t m{0};
  std::uint64_t k{0};
  HipblasLtPlanDataType data_type{HipblasLtPlanDataType::kBfloat16};
  std::int32_t algorithm_id{-1};
  std::vector<std::uint8_t> algorithm_blob;
  std::uint64_t workspace_bytes{0};
  std::uint64_t median_nanoseconds{0};
  std::string solution_name;
  std::string kernel_name;

  [[nodiscard]] bool operator==(const HipblasLtPlanRecord&) const = default;
};

struct HipblasLtPlanDatabase {
  std::uint32_t schema_version{kHipblasLtPlanDatabaseSchemaVersion};
  HipblasLtPlanDatabaseKey key;
  std::vector<HipblasLtPlanRecord> records;
};

enum class HipblasLtPlanDatabaseLoadStatus : std::uint8_t {
  kLoaded,
  kNotFound,
  kInvalid,
  kIncompatible,
};

struct HipblasLtPlanDatabaseLoadResult {
  HipblasLtPlanDatabaseLoadStatus status{
      HipblasLtPlanDatabaseLoadStatus::kInvalid};
  HipblasLtPlanDatabase database;
  std::string error;
};

[[nodiscard]] HipblasLtPlanDatabaseLoadResult LoadHipblasLtPlanDatabase(
    const std::filesystem::path& path,
    const HipblasLtPlanDatabaseKey& expected_key);

[[nodiscard]] HipblasLtPlanDatabaseLoadResult InspectHipblasLtPlanDatabase(
    const std::filesystem::path& path);

[[nodiscard]] bool SaveHipblasLtPlanDatabase(
    const std::filesystem::path& path, const HipblasLtPlanDatabase& database,
    std::string* error);

[[nodiscard]] const HipblasLtPlanRecord* FindHipblasLtPlanRecord(
    const HipblasLtPlanDatabase& database, std::size_t batch_size,
    std::size_t m, std::size_t k, HipblasLtPlanDataType data_type);

}  // namespace gufo::hip::detail

#endif  // GUFO_CORE_HIP_DETAIL_HIPBLASLT_PLAN_DATABASE_HPP_
