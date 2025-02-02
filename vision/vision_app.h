#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <optional>

using namespace std::chrono_literals;

struct Detection {
  cv::Rect box;
  int class_id;
  float confidence;
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

using detections_t = std::vector<Detection>;
detections_t postprocessDetections(const std::vector<cv::Mat> &outs, cv::Mat *frame, const std::chrono::time_point<std::chrono::steady_clock> &stamp);

class VisionApp {
  // raw frame queue
  cv::VideoCapture m_cap;
  std::mutex m_raw_mutex;
  std::condition_variable m_raw_frame_available;
  std::deque<std::unique_ptr<cv::Mat>> m_raw_frame_queue;
  constexpr static size_t m_raw_frame_queue_size = 60;

  // last available frame for object detection
  std::condition_variable m_raw_frame_available_peek;
  std::mutex m_raw_mutex_peek;
  std::unique_ptr<cv::Mat> m_raw_frame_peek;

  // detections queue
  std::mutex m_detect_mutex;
  std::deque<detections_t> m_detections_queue;
  std::condition_variable m_detection_done;
  constexpr static size_t m_detections_size = 15;

  // network for inference
  cv::dnn::Net m_net;

  // running flag
  std::atomic_bool m_running = true;

  // stats
  std::atomic<float> m_detect_fps = 0.0f;
  std::atomic<float> m_capture_fps = 0.0f;

  std::optional<detections_t> get_detections();

public:
  VisionApp();
  virtual ~VisionApp();

  void detectAnnotateThread();
  void captureThread();
  void displayThread();
};