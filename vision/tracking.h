#pragma once
#include "detection.h"
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>
#include <vector>


using namespace std::chrono_literals;

struct Measurement {
  Eigen::Vector2f m_position;
  Eigen::Vector2f m_velocity;
  float m_confidence;
  std::chrono::time_point<std::chrono::steady_clock> m_timestamp;
};

struct Object {
  static size_t m_id_counter;
  Eigen::Vector2f m_position; // position estimate
  Eigen::Vector2f m_velocity; // velocity estimate
  Eigen::Vector2f m_size;
  int m_class_id;
  const size_t m_id = m_id_counter++;
  std::chrono::time_point<std::chrono::steady_clock> m_first_seen;
  std::deque<Measurement> m_measurements;
  static constexpr size_t m_max_measurements = 10;

  explicit Object(const Detection &detection)
      : m_position(Eigen::Vector2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2))
      , m_velocity(Eigen::Vector2f(0, 0))
      , m_size(Eigen::Vector2f(detection.box.width, detection.box.height))
      , m_class_id(detection.class_id)
      , m_first_seen(detection.timestamp) {
    m_measurements.emplace_back(m_position, m_velocity, detection.confidence, detection.timestamp);
  }
  void addMeasurement(const Measurement &measurement);
  void updateEstimate(std::chrono::milliseconds prediction_interval);
  bool lastSeenRecently(const std::chrono::time_point<std::chrono::steady_clock> &now, std::chrono::milliseconds max_age) const;
  std::chrono::milliseconds age(const std::chrono::time_point<std::chrono::steady_clock> &now) const;
};

class SingleClassTracking {
  std::function<std::vector<Detection>(const std::vector<int> &, const std::vector<int> &, const std::vector<float> &)> m_get_detections;
  std::list<std::unique_ptr<Object>> m_objects;
  const int m_class_id;
  constexpr static float m_initial_confidence = 0.6f;
  constexpr static float m_persistence_confidence = 0.3f;
  constexpr static auto m_max_age = 2s;
  constexpr static auto m_prediction_interval = 10ms;
  std::atomic_bool m_running = {true};
  mutable std::mutex m_objects_mutex;

public:
  explicit SingleClassTracking(
      int class_id, const std::function<std::vector<Detection>(const std::vector<int> &, const std::vector<int> &, const std::vector<float> &)> &get_detections)
      : m_get_detections(get_detections)
      , m_class_id(class_id) {
  }
  virtual ~SingleClassTracking() {
    stop();
  }

  bool detectionIsAssociatedWithObject(const Detection &detection, const Object &object) const;
  Object *associateDetectionWithObject(const Detection &detection) const;
  void checkForNewObjects(const std::vector<Detection> &detections);
  void checkForDeadObjects();
  void addMeasurements(const std::vector<Detection> &detections);
  void updateObjectEstimates();
  void updateLoop();
  void stop();
  const std::vector<Detection> getDetectionsFromObjects() const;
};