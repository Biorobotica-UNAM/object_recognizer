#include <rclcpp/rclcpp.hpp>
#include <detector_interfaces/srv/get_bbox.hpp>
#include <detector_interfaces/srv/bbox_to_pose.hpp>
#include <detector_interfaces/srv/detect_by_class.hpp>
#include <memory>

using detector_interfaces::srv::GetBbox;
using detector_interfaces::srv::BboxToPose;
using detector_interfaces::srv::DetectByClass;

class DetectFusionNode : public rclcpp::Node
{
public:
  DetectFusionNode() : Node("detect_fusion_node")
  {
    declare_parameter<double>("timeout_sec", 1.0);
    timeout_ = std::chrono::duration<double>(get_parameter("timeout_sec").as_double());

    yolo_srv_name_ = "/get_bbox";
    pc_srv_name_   = "/bbox_to_pose";

    yolo_client_ = create_client<GetBbox>(yolo_srv_name_);
    pc_client_   = create_client<BboxToPose>(pc_srv_name_);

    srv_ = create_service<DetectByClass>(
      "detect_by_class",
      std::bind(&DetectFusionNode::handle, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "Fusion async service ready. Calls %s and %s",
                yolo_srv_name_.c_str(), pc_srv_name_.c_str());
  }

private:
  void handle(const std::shared_ptr<DetectByClass::Request> req,
              std::shared_ptr<DetectByClass::Response> res)
  {
    res->success = false;

    // Espera a que los servicios existan (async no bloqueante)
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

    // Async YOLO call
    auto y_future = yolo_client_->async_send_request(
      yreq,
      [this, res, req](rclcpp::Client<GetBbox>::SharedFuture yres_future) {
        auto yres = yres_future.get();
        if (!yres->success) {
          RCLCPP_WARN(get_logger(), "YOLO failed");
          return;
        }

        // Prepara request para PC
        auto preq = std::make_shared<BboxToPose::Request>();
        preq->header = yres->header;
        preq->x1 = yres->x1;
        preq->y1 = yres->y1;
        preq->x2 = yres->x2;
        preq->y2 = yres->y2;

        RCLCPP_INFO(get_logger(), "YOLO detected bbox (%d,%d)-(%d,%d) for class %s",
                        preq->x1, preq->y1, preq->x2, preq->y2,
                        req->class_name.c_str());

        // Async PC call
        auto p_future = pc_client_->async_send_request(
          preq,
          [res, this](rclcpp::Client<BboxToPose>::SharedFuture pres_future) {
            auto pres = pres_future.get();
            if (!pres->success) {
              RCLCPP_WARN(get_logger(), "PC service failed");
              return;
            }
            

              RCLCPP_WARN(get_logger(), "PC wuuuuu   failed");

            // Devuelve respuesta
            res->pose = pres->pose;
            res->success = true;
            RCLCPP_INFO(get_logger(), "Detection fusion succeeded");
            RCLCPP_INFO(get_logger(), "Object pose: (%.2f, %.2f, %.2f)",
                        pres->pose.pose.position.x,
                        pres->pose.pose.position.y,
                        pres->pose.pose.position.z);
          });
      });
  }

  std::string yolo_srv_name_;
  std::string pc_srv_name_;
  std::chrono::duration<double> timeout_;

  rclcpp::Client<GetBbox>::SharedPtr yolo_client_;
  rclcpp::Client<BboxToPose>::SharedPtr pc_client_;
  rclcpp::Service<DetectByClass>::SharedPtr srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DetectFusionNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
