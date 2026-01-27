#include <rclcpp/rclcpp.hpp>
#include <detector_interfaces/srv/get_bbox.hpp>
#include <detector_interfaces/srv/bbox_to_pose.hpp>
#include <detector_interfaces/srv/detect_by_class.hpp>
#include <chrono>

using detector_interfaces::srv::GetBbox;
using detector_interfaces::srv::BboxToPose;
using detector_interfaces::srv::DetectByClass;

class DetectFusionNode : public rclcpp::Node {
public:
  DetectFusionNode() : Node("detect_fusion_node") {
    
    declare_parameter<double>("timeout_sec", 1.0);
      timeout_ = std::chrono::duration<double>(
      get_parameter("timeout_sec").as_double());

    yolo_srv_name_ = "/get_bbox";
    pc_srv_name_   = "/bbox_to_pose";

    yolo_client_ = create_client<GetBbox>(yolo_srv_name_);
    pc_client_   = create_client<BboxToPose>(pc_srv_name_);

    srv_ = create_service<DetectByClass>(
      "detect_by_class",
      std::bind(&DetectFusionNode::handle, this, std::placeholders::_1, std::placeholders::_2)
    );

    RCLCPP_INFO(get_logger(), "Fusion ready. calls %s and %s",
                yolo_srv_name_.c_str(), pc_srv_name_.c_str());
  }

private:
  void handle(const std::shared_ptr<DetectByClass::Request> req,
              std::shared_ptr<DetectByClass::Response> res) {
    res->success = false;

    if (!yolo_client_->wait_for_service(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_logger(), "YOLO service not available");
      return;
    }
    if (!pc_client_->wait_for_service(std::chrono::milliseconds(500))) {
      RCLCPP_WARN(get_logger(), "PC service not available");
      return;
    }

    auto yreq = std::make_shared<GetBbox::Request>();
    yreq->class_name = req->class_name;

    auto y_future = yolo_client_->async_send_request(yreq);
    if (rclcpp::spin_until_future_complete(shared_from_this(), y_future, timeout_)
        != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "YOLO call timeout/fail");
      return;
    }

    auto yres = y_future.get();
    if (!yres->success) {
      return;
    }

    auto preq = std::make_shared<BboxToPose::Request>();
    preq->header = yres->header;
    preq->x1 = yres->x1; preq->y1 = yres->y1; preq->x2 = yres->x2; preq->y2 = yres->y2;

    auto p_future = pc_client_->async_send_request(preq);
    if (rclcpp::spin_until_future_complete(shared_from_this(), p_future, timeout_)
        != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_WARN(get_logger(), "PC call timeout/fail");
      return;
    }

    auto pres = p_future.get();
    if (!pres->success) {
      return;
    }

    res->pose = pres->pose;
    res->success = true;
  }

  std::string yolo_srv_name_;
  std::string pc_srv_name_;
  std::chrono::duration<double> timeout_;

  rclcpp::Client<GetBbox>::SharedPtr yolo_client_;
  rclcpp::Client<BboxToPose>::SharedPtr pc_client_;
  rclcpp::Service<DetectByClass>::SharedPtr srv_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectFusionNode>());
  rclcpp::shutdown();
  return 0;
}
