#ifndef PYTHON_VECTOR_STORE_H
#define PYTHON_VECTOR_STORE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// Forward declarations
class VectorClusterStore;
class Logger;

// Simple wrapper class for Python that hides implementation details
class PyVectorStore {
private:
    std::unique_ptr<VectorClusterStore> store_;
    std::unique_ptr<Logger> logger_;
    std::string device_path_;
    uint32_t vector_dim_;

public:
    PyVectorStore(const std::string& device_path, uint32_t vector_dim);
    ~PyVectorStore();
    
    bool store_vector(uint32_t id, const std::vector<float>& vector, const std::string& metadata = "");
    std::vector<float> get_vector(uint32_t id);
    std::vector<std::pair<uint32_t, float>> find_nearest(const std::vector<float>& query, uint32_t k = 10);
    bool delete_vector(uint32_t id);
    bool perform_maintenance();
    std::string get_metadata(uint32_t id) const;
    
    const std::string& get_device_path() const { return device_path_; }
    uint32_t get_vector_dim() const { return vector_dim_; }
};

#endif // PYTHON_VECTOR_STORE_H
