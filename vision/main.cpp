#include "tracking.h"
#include "vision_app.h"
#include <argparse/argparse.hpp>
#include <iostream>

int main(int argc, char **argv) {
  argparse::ArgumentParser program("vision");
  program.add_argument("-c", "--model-cfg").help("Path to the model config file (app is tuned for YOLOv3)").required();
  program.add_argument("-w", "--model-weights").help("Path to the model weights file (app is tuned for YOLOv3)").required();

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }

  VisionApp app(program.get<std::string>("--model-cfg"), program.get<std::string>("--model-weights"));

  SingleClassTracking person_tracker(
      0, std::bind(&VisionApp::filter_recent_detections, &app, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

  app.m_object_source = std::bind(&SingleClassTracking::getDetectionsFromObjects, &person_tracker);

  std::thread tracking_thread(&SingleClassTracking::updateLoop, &person_tracker);

  app.wait();
  person_tracker.stop();

  tracking_thread.join();
  return 0;
}
