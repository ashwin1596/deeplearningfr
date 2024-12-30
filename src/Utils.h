//
// Created by ashwi on 11/18/2024.
//

#ifndef UTILS_H
#define UTILS_H


#include <vector>
#include <iostream>
#include <mutex>
#include <memory>
#include <cstdlib>
#include <string>

std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& vec);
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<std::vector<size_t>>& vec_ptr);
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<std::vector<float>>& vec_ptr);

enum class DeviceType {
	CPU,
	GPU
};

class DeviceManager{
public:
	static DeviceType getDeviceType();

	static DeviceType parseDeviceType(std::string const device);

	static std::string deviceTypeToString(DeviceType device);
};

class Config {
public:
    static Config& getInstance();

    void loadFromEnv();

    // Getters
    DeviceType getDeviceType() const;
    std::string getCudaDevices() const;
    size_t getBatchSize() const;
    int getNumEpochs() const;

    // Setters
    void setDeviceType(const std::string& type);
    void setCudaDevices(const std::string& devices);
    void setBatchSize(size_t size);
    void setNumEpochs(int epochs);

private:
    Config();

    DeviceType device_type = DeviceType::CPU;  // default value
    std::string cuda_visible_devices = "0";  // default value
    size_t batch_size = 32;  // default value
    int num_epochs = 10;
};
#endif //UTILS_H
