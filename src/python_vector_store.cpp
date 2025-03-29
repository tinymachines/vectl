#include "python_vector_store.h"
#include "vector_cluster_store.h"
#include "logger.h"

PyVectorStore::PyVectorStore(const std::string& device_path, uint32_t vector_dim)
    : device_path_(device_path), vector_dim_(vector_dim) {
    logger_ = std::make_unique<Logger>("vector_store_python.log");
    store_ = std::make_unique<VectorClusterStore>(*logger_);
    store_->initialize(device_path, "kmeans", vector_dim);
}

PyVectorStore::~PyVectorStore() = default;

bool PyVectorStore::store_vector(uint32_t id, const std::vector<float>& vector, const std::string& metadata) {
    return store_->storeVector(id, vector, metadata);
}

std::vector<float> PyVectorStore::get_vector(uint32_t id) {
    std::vector<float> result(vector_dim_);
    store_->retrieveVector(id, result);
    return result;
}

std::vector<std::pair<uint32_t, float>> PyVectorStore::find_nearest(const std::vector<float>& query, uint32_t k) {
    return store_->findSimilarVectors(query, k);
}

bool PyVectorStore::delete_vector(uint32_t id) {
    return store_->deleteVector(id);
}

bool PyVectorStore::perform_maintenance() {
    return store_->performMaintenance();
}

std::string PyVectorStore::get_metadata(uint32_t id) const {
    // This is a stub - implement properly based on your metadata storage
    return "Metadata for vector " + std::to_string(id);
}
