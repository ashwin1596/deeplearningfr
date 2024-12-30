//
// Created by ashwi on 11/12/2024.
//

#ifndef DATASET_H
#define DATASET_H
#include <utility>
#include "Tensor.cuh"
#include <cstdint>


class Dataset {
public:
  virtual size_t size() const = 0;
  virtual std::pair<TensorPtr, TensorPtr> get(size_t idx) = 0;
  virtual ~Dataset() = default;
  virtual void reload() = 0;
};


class MNISTDataset : public Dataset{
public:
  MNISTDataset(const std::string& file_path);
  size_t size() const override;
  std::pair<TensorPtr, TensorPtr> get(size_t idx) override;
  void reload() override;
private:
  void loadLibSVMData(const std::string& data_file);

  std::vector<TensorPtr> images_;
  std::vector<TensorPtr> labels_;
  std::string file_path;
};

#endif //DATASET_H
