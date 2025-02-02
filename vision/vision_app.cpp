#include "vision_app.h"
#include <iostream>
#include <thread>

detections_t postprocessDetections(const std::vector<cv::Mat> &outs, cv::Mat *frame, const std::chrono::time_point<std::chrono::steady_clock> &stamp) {
  std::vector<float> confs;
  std::vector<cv::Rect> boxes;
  std::vector<int> classIds;
  for (size_t i = 0; i < outs.size(); ++i) {
    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.
    float *data = (float *)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point classIdPoint;
      double confidence;
      // Get the value and location of the maximum score
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > 0.5) {
        int centerX = (int)(data[0] * frame->cols);
        int centerY = (int)(data[1] * frame->rows);
        int width = (int)(data[2] * frame->cols);
        int height = (int)(data[3] * frame->rows);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        boxes.emplace_back(left, top, width, height);
        confs.push_back(confidence);
        classIds.push_back(classIdPoint.x);
      }
    }
  }
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confs, 0.5, 0.4, indices);
  std::vector<Detection> detections;
  for (auto index : indices) {
    detections.push_back({boxes[index], classIds[index], confs[index], stamp});
  }
  return detections;
}

std::optional<detections_t> VisionApp::get_detections() {
  auto now = std::chrono::high_resolution_clock::now();
  std::optional<detections_t> detections;
  {
    std::unique_lock<std::mutex> lock(m_detect_mutex);
    auto detections_available = m_detection_done.wait_for(lock, 1us, [this] { return !m_detections_queue.empty(); });
    if (detections_available) {
      detections = m_detections_queue.front();
      m_detections_queue.pop_front();
    }
  }
  return detections;
}

VisionApp::VisionApp() {
  m_cap.open(0);
  if (!m_cap.isOpened()) {
    std::cerr << "Error: Could not open the camera!" << std::endl;
    return;
  }
  try {
    // m_net = cv::dnn::readNetFromDarknet("D:/Devel/vision_example/yolov3-tiny.cfg", "D:/Devel/vision_example/yolov3-tiny.weights");
    m_net = cv::dnn::readNetFromDarknet("D:/Devel/vision_example/yolov3.cfg", "D:/Devel/vision_example/yolov3.weights");
  } catch (cv::Exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return;
  }
  m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

  std::thread cap_thread(&VisionApp::captureThread, this);
  std::thread detect_thread(&VisionApp::detectAnnotateThread, this);
  std::thread display_thread(&VisionApp::displayThread, this);

  cap_thread.join();
  detect_thread.join();
  display_thread.join();
}

VisionApp::~VisionApp() {
  if (m_cap.isOpened())
    m_cap.release();
  cv::destroyAllWindows();
}

void VisionApp::detectAnnotateThread() {
  try {
    while (m_running) {
      auto detect_begin = std::chrono::high_resolution_clock::now();
      std::unique_ptr<cv::Mat> frame;
      {
        std::unique_lock<std::mutex> lock(m_raw_mutex_peek);
        auto frame_available = m_raw_frame_available_peek.wait_for(lock, 1us, [this] { return bool(m_raw_frame_peek); });
        if (frame_available) {
          frame = std::move(m_raw_frame_peek);
          m_raw_frame_peek.reset();
        } else {
          continue;
        }
      }

      // inference
      cv::Mat blob = cv::dnn::blobFromImage(*frame, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
      m_net.setInput(blob);
      std::vector<cv::Mat> outs;
      m_net.forward(outs, m_net.getUnconnectedOutLayersNames());

      // postprocess detections
      auto detections = postprocessDetections(outs, frame.get(), detect_begin);

      {
        std::unique_lock<std::mutex> lock(m_detect_mutex);
        // remove detections from the queue if full
        if (m_detections_queue.size() > m_detections_size) {
          m_detections_queue.pop_front();
        }
        m_detections_queue.push_back(std::move(detections));
      }
      m_detection_done.notify_one();

      auto detect_end = std::chrono::high_resolution_clock::now();
      auto detect_fps = 1.0f / std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_begin).count() * 1000.0f;
      auto this_duration = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_begin);
      m_detect_fps = detect_fps;
    }
  } catch (cv::Exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Error: Unknown exception!" << std::endl;
  }
}

void VisionApp::captureThread() {
  while (m_running) {
    auto cap_begin = std::chrono::high_resolution_clock::now();

    // perform capture
    auto frame = std::make_unique<cv::Mat>();
    try {
      m_cap.read(*frame);

      if (frame->empty()) {
        std::cerr << "Error: Could not grab a frame!" << std::endl;
      }
    } catch (cv::Exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }

    // populating the raw frame queue
    {
      std::scoped_lock lock(m_raw_mutex, m_raw_mutex_peek);
      // remove the oldest frame if the queue is full
      if (m_raw_frame_queue.size() >= m_raw_frame_queue_size) {
        m_raw_frame_queue.pop_front();
      }
      m_raw_frame_peek = std::make_unique<cv::Mat>(*frame);
      m_raw_frame_queue.push_back(std::move(frame));
    }
    m_raw_frame_available.notify_all();
    m_raw_frame_available_peek.notify_all();

    auto cap_end = std::chrono::high_resolution_clock::now();
    auto this_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cap_end - cap_begin);
    auto cap_fps = 1.0f / this_duration.count() * 1000.0f;
    m_capture_fps = cap_fps;
  }
}

void VisionApp::displayThread() {
  while (true) {
    auto display_begin = std::chrono::high_resolution_clock::now();
    // pull the frame off the queue
    std::unique_ptr<cv::Mat> frame;
    {
      std::unique_lock<std::mutex> lock(m_raw_mutex);
      m_raw_frame_available.wait(lock, [this] { return !m_raw_frame_queue.empty(); });
      frame = std::move(m_raw_frame_queue.front());
      m_raw_frame_queue.pop_front();
    }

    // draw the detections, if any
    auto detections = get_detections();
    if (detections) {
      for (const auto &detection : *detections) {
        cv::rectangle(*frame, detection.box, cv::Scalar(0, 255, 0), 2);
        cv::putText(*frame, std::to_string(detection.class_id), cv::Point(detection.box.x, detection.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
      }
    }

    // cv::flip(*frame, *frame, 1);

    // print the detection fps
    cv::putText(*frame, "FPS: " + std::to_string(m_detect_fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    // draw the frame
    cv::imshow("Webcam Feed", *frame);

    if (cv::waitKey(1) == 27) {
      m_running = false;
      break;
    }
  }
}