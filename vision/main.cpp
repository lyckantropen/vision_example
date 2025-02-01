#include <atomic>
#include <chrono>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

using namespace std::chrono_literals;

struct Detection {
  cv::Rect box;
  int class_id;
  float confidence;
};

class VisionApp {
  std::mutex m_raw_mutex;
  std::mutex m_annotation_mutex;
  std::condition_variable m_raw_frame_available;
  std::condition_variable m_annotated_frame_available;
  std::queue<std::unique_ptr<cv::Mat>> m_raw_frame_queue;
  std::queue<std::unique_ptr<cv::Mat>> m_annotated_frame_queue;
  constexpr static size_t m_frame_queue_size = 60;

  cv::VideoCapture m_cap;
  std::atomic_bool m_running = true;
  cv::dnn::Net m_net;
  std::vector<Detection> m_detections;

public:
  VisionApp() {
    m_cap.open(0);
    if (!m_cap.isOpened()) {
      std::cerr << "Error: Could not open the camera!" << std::endl;
      return;
    }
    try {
      //m_net = cv::dnn::readNetFromONNX("D:/Devel/vision_example/tinyyolov2-7  .onnx");
      m_net = cv::dnn::readNetFromDarknet("D:/Devel/vision_example/yolov3-tiny.cfg", "D:/Devel/vision_example/yolov3-tiny.weights");
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
  virtual ~VisionApp() {
    if (m_cap.isOpened())
      m_cap.release();
    cv::destroyAllWindows();
  }

  void detectAnnotateThread() {
    try {
      while (m_running) {
        auto frame = [this]() {
          std::unique_lock<std::mutex> lock(m_raw_mutex);
          m_raw_frame_available.wait(lock, [this] { return !m_raw_frame_queue.empty(); });
          auto frame = std::move(m_raw_frame_queue.front());
          m_raw_frame_queue.pop();
          return frame;
        }();

        auto begin = std::chrono::high_resolution_clock::now();
        cv::Mat blob = cv::dnn::blobFromImage(*frame, 1.0/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        m_net.setInput(blob);
        std::vector<cv::Mat> outs;
        m_net.forward(outs, m_net.getUnconnectedOutLayersNames());

        m_detections.clear();
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Scan through all the bounding boxes output from the network and keep only the
            // ones with high confidence scores. Assign the box's class label as the class
            // with the highest score for the box.
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > 0.5)
                {
                    int centerX = (int)(data[0] * frame->cols);
                    int centerY = (int)(data[1] * frame->rows);
                    int width = (int)(data[2] * frame->cols);
                    int height = (int)(data[3] * frame->rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    m_detections.push_back({cv::Rect(left, top, width, height), classIdPoint.x, (float)confidence});
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Inference time: " << duration.count() << "ms" << std::endl;
        std::cout << "Number of detected objects: " << m_detections.size() << std::endl;

        {
          std::unique_lock<std::mutex> lock(m_annotation_mutex);
          m_annotated_frame_queue.push(std::move(frame));
        }
        m_annotated_frame_available.notify_one();
      }
    } catch (cv::Exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
    } catch (std::exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "Error: Unknown exception!" << std::endl;
    }
  }

  void captureThread() {
    while (m_running) {
      auto cap_begin = std::chrono::high_resolution_clock::now();

      auto frame = std::make_unique<cv::Mat>();
      try {
        m_cap.read(*frame);

        if (frame->empty()) {
          std::cerr << "Error: Could not grab a frame!" << std::endl;
        }
      } catch (cv::Exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
      }

      {
        std::lock_guard<std::mutex> lock(m_raw_mutex);
        // remove the oldest frame if the queue is full
        if (m_raw_frame_queue.size() >= m_frame_queue_size) {
          m_raw_frame_queue.pop();
        }
        m_raw_frame_queue.push(std::move(frame));
      }

      m_raw_frame_available.notify_one();

      // sleep for as long as it takes to stick to 60fps
      // auto cap_end = std::chrono::high_resolution_clock::now();
      // auto cap_duration =
      // std::chrono::duration_cast<std::chrono::milliseconds>(
      //     cap_end - cap_begin);
      // std::this_thread::sleep_for(
      //     std::max(0ms, std::chrono::milliseconds(16) - cap_duration));
    }
  }
  void displayThread() {
    while (true) {
      cv::Mat tmp_frame;
      {
        std::unique_lock<std::mutex> lock(m_annotation_mutex);
        m_annotated_frame_available.wait(lock, [this] { return !m_annotated_frame_queue.empty(); });
        // tmp_frame = ;
        cv::flip(*m_annotated_frame_queue.front(), tmp_frame, 1);
        m_annotated_frame_queue.pop();
      }

      cv::imshow("Webcam Feed", tmp_frame);

      if (cv::waitKey(1) == 27) {
        m_running = false;
        break;
      }
    }
  }
};

int main() {
  VisionApp app;
  return 0;
}
