//
// Created by ashwi on 11/18/2024.
//
#include "Utils.h"


// Overload the << operator for std::vector<size_t>
std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& vec) {
	if(vec.empty()) {
		os << "Scalar";
		return os;
	}

	os << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		os << vec[i];
		if (i != vec.size() - 1) {
			os << ", ";
		}
	}
	os << "]";
	return os;
}

// Overload the << operator for std::vector<size_t>
std::ostream& operator<<(std::ostream& os, const std::vector<float>& vec) {
	if(vec.empty()) {
		os << "Scalar";
		return os;
	}

	os << "[";
	for (size_t i = 0; i < vec.size(); ++i) {
		os << vec[i];
		if (i != vec.size() - 1) {
			os << ", ";
		}
	}
	os << "]";
	return os;
}

// Overload the << operator for std::shared_ptr<std::vector<size_t>>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<std::vector<size_t>>& vec_ptr) {
	if (!vec_ptr) {
        os << "nullptr"; // Handle null shared_ptr
        return os;
    }

    const auto& vec = *vec_ptr; // Dereference the shared_ptr
    if (vec.empty()) {
        os << "Scalar"; // Handle empty vector case
        return os;
    }

    os << "["; // Start of the vector
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i]; // Output each element
        if (i != vec.size() - 1) {
            os << ", "; // Separator for elements
        }
    }
    os << "]"; // End of the vector
    return os;
}

// Overload the << operator for std::shared_ptr<std::vector<float>>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<std::vector<float>>& vec_ptr) {
	if (!vec_ptr) {
        os << "nullptr"; // Handle null shared_ptr
        return os;
    }

    const auto& vec = *vec_ptr; // Dereference the shared_ptr
    if (vec.empty()) {
        os << "Scalar"; // Handle empty vector case
        return os;
    }

    os << "["; // Start of the vector
    for (size_t i = 0; i < vec.size(); ++i) {
        if(vec[i] !=0 )
        {
            os << vec[i]; // Output each element
            if (i != vec.size() - 1) {
                os << ", "; // Separator for elements
            }
        }
    }
    os << "]"; // End of the vector
    return os;
}


std::string DeviceManager::deviceTypeToString(DeviceType device) {
    switch (device) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::GPU: return "GPU";
        default: return "Unknown";
    }
}

DeviceType DeviceManager::getDeviceType(){
        const char* device = std::getenv("DEVICE_TYPE");
		static DeviceType currentDevice = parseDeviceType(std::string(device));
		return currentDevice;
}

DeviceType DeviceManager::parseDeviceType(std::string const device){

	if(device == "GPU" || device == "gpu"){
		return DeviceType::GPU;
	}
	return DeviceType::CPU;
}


    Config& Config::getInstance() {
        static Config instance;
        return instance;
    }

    void Config::loadFromEnv() {
        // Get device type
        const char* device = std::getenv("DEVICE_TYPE");
        if(device) device_type = DeviceManager::parseDeviceType(std::string(device));

        // Get CUDA device
        const char* cuda_device = std::getenv("CUDA_VISIBLE_DEVICES");
        if(cuda_device) cuda_visible_devices = std::string(cuda_device);

        // Get batch size
        const char* batch = std::getenv("BATCH_SIZE");
        if(batch) batch_size = std::stoull(batch);

        // Get number of epochs
        const char* epochs = std::getenv("NUM_EPOCHS");
        if(batch) num_epochs = std::stoull(epochs);
    }

    // Getters
    DeviceType Config::getDeviceType() const { return device_type; }
    std::string Config::getCudaDevices() const { return cuda_visible_devices; }
    size_t Config::getBatchSize() const { return batch_size; }
    int Config::getNumEpochs() const { return num_epochs; }

    // Setters
    void Config::setDeviceType(const std::string& type) { device_type = DeviceManager::parseDeviceType(type); }
    void Config::setCudaDevices(const std::string& devices) { cuda_visible_devices = devices; }
    void Config::setBatchSize(size_t size) { batch_size = size; }
    void Config::setNumEpochs(int epochs) { num_epochs = epochs; }

    Config::Config() {
        loadFromEnv();
    }
