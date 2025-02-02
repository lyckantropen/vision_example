#include <iostream>
#include "vision_app.h"

struct Object {
  cv::KalmanFilter kf;
  cv::Rect box;
  int class_id;
  float confidence;
  int missed_frames;
};

int main() {
  VisionApp app;
  return 0;
}
