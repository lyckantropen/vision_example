#include "tracking.h"

size_t Object::m_id_counter = 0;

void Object::addMeasurement(const Measurement &measurement) {
  if (m_measurements.size() >= m_max_measurements) {
    m_measurements.pop_front();
  }
  m_measurements.push_back(measurement);
}

void Object::updateEstimate(std::chrono::milliseconds prediction_interval) {
  // simple linear prediction
  if (m_measurements.size() > 1) {
    auto &last_measurement = m_measurements.back();
    auto &prev_measurement = m_measurements.at(m_measurements.size() - 2);
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(last_measurement.m_timestamp - prev_measurement.m_timestamp).count() / 1000.0f;
    dt = std::max(dt, std::chrono::duration_cast<std::chrono::milliseconds>(prediction_interval).count() / 1000.0f); // avoid division by zero
    m_velocity = (last_measurement.m_position - prev_measurement.m_position) / dt;
    m_position = last_measurement.m_position + m_velocity * dt;
  } else {
    // no velocity estimate
    m_position = m_measurements.back().m_position;
  }
  if (m_measurements.size()) {
    // set size to average of last measurements
    m_size = Eigen::Vector2f(0, 0);
    for (const auto &measurement : m_measurements) {
      m_size += measurement.m_position;
    }
    m_size /= m_measurements.size();
  }
}

bool Object::lastSeenRecently(const std::chrono::time_point<std::chrono::steady_clock> &now, std::chrono::milliseconds max_age) const {
  return now - m_measurements.back().m_timestamp < max_age;
}

std::chrono::milliseconds Object::age(const std::chrono::time_point<std::chrono::steady_clock> &now) const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(now - m_first_seen);
}

bool SingleClassTracking::detectionIsAssociatedWithObject(const Detection &detection, const Object &object) const {
  // simple distance-based association
  auto contained_in_box = detection.box.contains({(int)object.m_position.x(), (int)object.m_position.y()});
  auto same_class = detection.class_id == object.m_class_id;
  auto matching_confidence = detection.confidence > m_persistence_confidence;

  if (!contained_in_box) {
    // std::cout << "Not contained in box" << std::endl;
    return false;
  }
  if (!same_class) {
    // std::cout << "Not same class" << std::endl;
    return false;
  }
  if (!matching_confidence) {
    // std::cout << "Not matching confidence" << std::endl;
    return false;
  }
  return true;
}

Object *SingleClassTracking::associateDetectionWithObject(const Detection &detection) const {
  std::vector<size_t> objects_already_associated;
  const auto existing_object = std::find_if(m_objects.cbegin(), m_objects.cend(), [&](const std::unique_ptr<Object> &object) {
    auto already_associated =
        std::find_if(objects_already_associated.begin(), objects_already_associated.end(), [&](size_t id) { return id == object->m_id; });
    return already_associated == objects_already_associated.end() && detectionIsAssociatedWithObject(detection, *object);
  });
  if (existing_object != m_objects.cend()) {
    objects_already_associated.push_back((*existing_object)->m_id);
    return (*existing_object).get();
  }
  return nullptr;
}

void SingleClassTracking::checkForNewObjects(const std::vector<Detection> &detections) {
  for (const auto &detection : detections) {
    if (detection.confidence < m_initial_confidence) {
      continue;
    }
    auto *existing_object = associateDetectionWithObject(detection);
    if (existing_object == nullptr) {
      // this is a new object
      std::unique_lock<std::mutex> lock(m_objects_mutex);
      m_objects.push_back(std::make_unique<Object>(detection));

      std::cout << "New object: " << detection.class_id << " at "
                << Eigen::Vector2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2).transpose() << std::endl;
    }
  }
}

void SingleClassTracking::checkForDeadObjects() {
  auto now = std::chrono::steady_clock::now();
  auto dead_object = [&](const std::unique_ptr<Object> &object) {
    auto should_remove = !object->lastSeenRecently(now, m_max_age);
    if (should_remove) {
      std::cout << "Removing object: " << object->m_class_id << " at " << object->m_position.transpose()
                << ", lived for: " << object->age(now).count() / 1000.0f << " seconds" << std::endl;
    }
    return should_remove;
  };

  std::unique_lock<std::mutex> lock(m_objects_mutex);
  m_objects.erase(std::remove_if(m_objects.begin(), m_objects.end(), dead_object), m_objects.end());
}

void SingleClassTracking::addMeasurements(const std::vector<Detection> &detections) {
  for (const auto &detection : detections) {
    auto existing_object = associateDetectionWithObject(detection);
    if (existing_object) {
      existing_object->addMeasurement({Eigen::Vector2f(detection.box.x + detection.box.width / 2, detection.box.y + detection.box.height / 2),
                                       Eigen::Vector2f(0, 0), detection.confidence, detection.timestamp});
    }
  }
}

void SingleClassTracking::updateObjectEstimates() {
  std::unique_lock<std::mutex> lock(m_objects_mutex);
  for (auto &object : m_objects) {
    object->updateEstimate(m_prediction_interval);
  }
}

void SingleClassTracking::updateLoop() {
  while (m_running) {
    auto detections = m_get_detections({m_class_id}, {1}, {m_persistence_confidence});

    checkForDeadObjects();
    // filter only detections higher than initial confidence
    checkForNewObjects(detections);
    addMeasurements(detections);
    updateObjectEstimates();
    std::this_thread::sleep_for(m_prediction_interval);

    // print object positions
    for (const auto &object : m_objects) {
      std::cout << "Object: " << object->m_id << ", " << object->m_class_id << " at (" << object->m_position.x() << ", " << object->m_position.y()
                << ") with velocity (" << object->m_velocity.x() << ", " << object->m_velocity.y() << ")" << std::endl;
    }
  }
}

void SingleClassTracking::stop() {
  m_running = false;
}

const std::vector<Detection> SingleClassTracking::getDetectionsFromObjects() const {
  std::unique_lock<std::mutex> lock(m_objects_mutex);
  std::vector<Detection> detections;
  std::transform(m_objects.begin(), m_objects.end(), std::back_inserter(detections), [](const std::unique_ptr<Object> &object) {
    return Detection{
        cv::Rect(object->m_position.x() - object->m_size.x() / 2, object->m_position.y() - object->m_size.y() / 2, object->m_size.x(), object->m_size.y()),
        object->m_class_id, 1.0f, std::chrono::steady_clock::now()};
  });
  return detections;
}