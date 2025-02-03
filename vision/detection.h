#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  int class_id;
  float confidence;
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
};