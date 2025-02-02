#include <iostream>
#include "vision_app.h"
#include <argparse/argparse.hpp>

struct Object {
  cv::KalmanFilter kf;
  cv::Rect box;
  int class_id;
  float confidence;
  int missed_frames;
};

int main(int argc, char** argv) {
  argparse::ArgumentParser program("vision");
  program.add_argument("-c", "--model-cfg")
    .help("Path to the model config file (app is tuned for YOLOv3)")
    .required();
  program.add_argument("-w", "--model-weights")
    .help("Path to the model weights file (app is tuned for YOLOv3)")
    .required();
  
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }

  VisionApp app(program.get<std::string>("--model-cfg"), program.get<std::string>("--model-weights"));
  return 0;
}
