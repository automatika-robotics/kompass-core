#include "datatypes/tracking.h"
#include "test.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include "vision/depth_detector.h"
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <memory>

using namespace Kompass;

struct DepthDetectorTestConfig {
  std::unique_ptr<DepthDetector> detector;
  Path::State current_state = {1.0, 1.0, 0.0};
  std::vector<Bbox2D> detected_boxes;
  std::string pltFileName = "DepthDetectorTest";
  Eigen::Vector2f focal_length = {911.71, 910.288};
  Eigen::Vector2f principal_point = {643.06, 366.72};
  Eigen::Vector2f depth_range = {0.001, 5.0}; // 5cm to 5 meters
  float depth_conv_factor = 1e-3;             // convert from mm to m
  Eigen::Isometry3f camera_body_tf;
  Eigen::MatrixX<unsigned short> depth_image;
  cv::Mat cv_img;
  std::vector<Bbox2D> detections;

  DepthDetectorTestConfig(const std::string image_filename, const Bbox2D &box) {
    // Body to camera tf from robot of test pictures
    auto link_in_body =
        getTransformation(Eigen::Quaternionf{0.0f, 0.1987f, 0.0f, 0.98f},
                          Eigen::Vector3f{0.32f, 0.0209f, 0.3f});

    auto cam_in_link =
        getTransformation(Eigen::Quaternionf{0.01f, -0.00131f, 0.002f, 0.9999f},
                          Eigen::Vector3f{0.0f, 0.0105f, 0.0f});

    auto cam_opt_in_cam =
        getTransformation(Eigen::Quaternionf{-0.5f, 0.5f, -0.5f, 0.5f},
                          Eigen::Vector3f{0.0f, 0.0105f, 0.0f});

    Eigen::Isometry3f cam_in_body = link_in_body * cam_in_link * cam_opt_in_cam;

    detector =
        std::make_unique<DepthDetector>(depth_range, cam_in_body, focal_length,
                                        principal_point, depth_conv_factor);

    cv_img = cv::imread(image_filename, cv::IMREAD_GRAYSCALE);

    if (cv_img.empty()) {
      LOG_ERROR("Could not open or find the image");
    }

    // Create an Eigen matrix of type int from the OpenCV Mat
    depth_image = Eigen::MatrixX<unsigned short>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; ++i) {
      for (int j = 0; j < cv_img.cols; ++j) {
        depth_image(i, j) = cv_img.at<unsigned short>(i, j);
      }
    }
    detections.push_back(box);
  };

  std::vector<Bbox3D> run(const std::string outputFilename,
                          const bool local_frame = true) {
    if(local_frame){
      detector->updateBoxes(depth_image, detections);
    }
    else{
      detector->updateBoxes(depth_image, detections, current_state);
    }

    auto boxes3D = detector->get3dDetections();
    if (boxes3D) {
      auto res = boxes3D.value();

      // Draw the bounding boxes on the image
      cv::Scalar color(0, 0, 0); // Green color for the rectangles
      int thickness = 2;         // Thickness of the rectangle lines

      for (const auto &box : res) {
        LOG_INFO("Got detected box in 3D world coordinates at :",
                 box.center.x(), ", ", box.center.y(), ", ", box.center.z());
        LOG_INFO("Box size :", box.size.x(), ", ", box.size.y(), ", ",
                 box.size.z());
        cv::Point topLeft(box.center_img_frame.x() - box.size_img_frame.x() / 2,
                          box.center_img_frame.y() -
                              box.size_img_frame.y() / 2);
        cv::Point bottomRight(
            box.center_img_frame.x() + box.size_img_frame.x() / 2,
            box.center_img_frame.y() + box.size_img_frame.y() / 2);
        cv::rectangle(cv_img, topLeft, bottomRight, color, thickness);
      }

      // Save the modified image
      cv::imwrite(outputFilename, cv_img);

      if (!cv::imwrite(outputFilename, cv_img)) {
        LOG_ERROR("Could not save the image");
      } else {
        LOG_INFO("Image saved to ", outputFilename);
      }
      BOOST_TEST(res.size() == 1, "Got different size for 3D and 2D boxes");
      BOOST_TEST(res[0].size_img_frame.x() == detections[0].size.x(),
                 "Error parsing box, size x in pixel frame is not conserved");
      BOOST_TEST(res[0].size_img_frame.y() == detections[0].size.y(),
                 "Error parsing box, size y in pixel frame is not conserved");
      return res;
    }
    throw std::runtime_error("Could not find 3D boxes");
  }
};

BOOST_AUTO_TEST_CASE(test_Depth_Detector_person_image) {
  // Create timer
  Timer time;
  std::string filename =
      "/home/ahr/kompass/uvmap_code/resources/depth_image.tif";
  Bbox2D box({535, 0}, {520, 420});
  auto config = DepthDetectorTestConfig(filename, box);
  std::string outputFilename =
      "/home/ahr/kompass/uvmap_code/resources/image_output.jpg";
  auto boxes = config.run(outputFilename);
}


BOOST_AUTO_TEST_CASE(test_Depth_Detector_bag_image_local_frame) {
  LOG_INFO("Testing and generating 3D boxes in local robot frame");
  // Create timer
  Timer time;
  std::string filename =
      "/home/ahr/kompass/uvmap_code/resources/bag_image_depth.tif";
  Bbox2D box({410, 0}, {410, 390});
  auto config = DepthDetectorTestConfig(filename, box);
  std::string outputFilename =
      "/home/ahr/kompass/uvmap_code/resources/bag_depth_output.jpg";
  auto boxes = config.run(outputFilename, true);
  float dist =
      std::sqrt(std::pow((boxes[0].center.x()), 2) +
                std::pow((boxes[0].center.y()), 2));
  const float approx_actual_dist = 1.8;
  BOOST_TEST(std::abs(dist - approx_actual_dist) <= 0.1,
             "3D box distance is not equal to approximate measured distance");
}

BOOST_AUTO_TEST_CASE(test_Depth_Detector_bag_image_global_frame) {
  LOG_INFO("Testing and generating 3D boxes in global world frame");
  // Create timer
  Timer time;
  std::string filename =
      "/home/ahr/kompass/uvmap_code/resources/bag_image_depth.tif";
  Bbox2D box({410, 0}, {410, 390});
  auto config = DepthDetectorTestConfig(filename, box);
  std::string outputFilename =
      "/home/ahr/kompass/uvmap_code/resources/bag_depth_output.jpg";
  auto boxes = config.run(outputFilename, false);
  float dist =
      std::sqrt(std::pow((boxes[0].center.x() - config.current_state.x), 2) +
                std::pow((boxes[0].center.y() - config.current_state.y), 2));
  const float approx_actual_dist = 1.8;
  BOOST_TEST(std::abs(dist - approx_actual_dist) <= 0.1,
             "3D box distance is not equal to approximate measured distance");
}
