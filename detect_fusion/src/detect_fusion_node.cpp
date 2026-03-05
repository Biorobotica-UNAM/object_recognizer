#include <chrono>
#include <memory>
#include <string>
#include <future>

#include "rclcpp/rclcpp.hpp"
#include "vision_interfaces/srv/get_bbox.hpp"
#include "vision_interfaces/srv/bbox_to_pose.hpp"
#include "vision_interfaces/srv/detect_by_class.hpp"
#include "vision_interfaces/srv/owlv2_service.hpp"

using vision_interfaces::srv::GetBbox;
using vision_interfaces::srv::BboxToPose;
using vision_interfaces::srv::DetectByClass;
using vision_interfaces::srv::Owlv2Service;


class DetectFusionNode
  : public rclcpp::Node
  , public std::enable_shared_from_this<DetectFusionNode>
{
public:
  DetectFusionNode() : Node("detect_fusion_node")
  {
    declare_parameter<double>("timeout_sec", 5.0);
    timeout_ = std::chrono::duration<double>(get_parameter("timeout_sec").as_double());

    detector_services_ = {
      {"YOLO", "/yolo/get_bbox"},
      {"OWL",  "/owl/Owlv2Service"},
      //{"DINO", "/dino/get_bbox"}
    };

    pc_srv_name_   = "/bbox_to_pose";

    // Callback groups:
    cbg_srv_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cbg_clients_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    //yolo_client_ = this->create_client<GetBbox>(yolo_srv_name_, rclcpp::ServicesQoS(), cbg_clients_);
    for (const auto& [model, service_name] : detector_services_){
      detectors_[model] = this->create_client<GetBbox>(service_name, rmw_qos_profile_services_default, cbg_clients_);
      RCLCPP_INFO(get_logger(), "Detector client for %s created, service: %s", model.c_str(), service_name.c_str());
    }

    pc_client_   = this->create_client<BboxToPose>(pc_srv_name_, rmw_qos_profile_services_default, cbg_clients_);

    //auto service_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    srv_ = this->create_service<DetectByClass>(
      "/detect_by_class",
      std::bind(&DetectFusionNode::handle, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, cbg_srv_);

    RCLCPP_INFO(get_logger(), "Detect fusion service ready: /detect_by_class");
  }

private:
 void handle(const std::shared_ptr<DetectByClass::Request> req,
              std::shared_ptr<DetectByClass::Response> res)
  {
    res->success = false;

    if(req->class_name.empty()) {
      res->message = "Empty class name";
      return;
    }

    auto it = detectors_.find(req->model);
    if (it == detectors_.end()) {
      res->message = "Unknown detector model: " + req->model;
      return;
    }
    auto detector_client_ptr = it->second;

    // 1. Espera a que existan los servicios
    if (!detector_client_ptr->wait_for_service(std::chrono::seconds(1))) {
      res->message = "Detector " + req->model + " service not available";
      return;
    }
    if (!pc_client_->wait_for_service(std::chrono::seconds(1))) {
      res->message = "PointCloud service not available";
      return;
    }

    // 2. Llamada al Detector (YOLO/OWL)
    auto dreq = std::make_shared<GetBbox::Request>();
    dreq->class_name = req->class_name;

    auto d_future = detector_client_ptr->async_send_request(dreq);

    auto d_status = d_future.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_));
    if (d_status != std::future_status::ready)
    {
      res->message = "Detector call timeout (Model: " + req->model + ")";
      return;
    }

    auto dres = d_future.get();
    if (!dres->success) {
      res->message = req->model + " returned success=false";
      return;
    }

    auto preq = std::make_shared<BboxToPose::Request>();
    
    // Seguridad para el Header
    if (dres->header.stamp.sec == 0) {
        RCLCPP_WARN(this->get_logger(), "Detector %s sent empty timestamp. Using current time.", req->model.c_str());
        preq->header.stamp = this->get_clock()->now();
        preq->header.frame_id = "/kinect/image_raw";
    } else {
        preq->header = dres->header;
    }

    // PointCloud service
    preq->header = dres->header;
    preq->x1 = dres->x1;
    preq->y1 = dres->y1;
    preq->x2 = dres->x2;
    preq->y2 = dres->y2;

    auto p_future = pc_client_->async_send_request(preq);

    auto p_status = p_future.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_));
    if (p_status != std::future_status::ready)
    {
      res->message = "PointCloud call timeout";
      return;
    }

    auto pres = p_future.get();
    if (!pres->success) {
      res->message = "PointCloud returned success=false";
      return;
    }

    res->pose = pres->pose;
    res->message = "Detection successful with " + req->model;
    res->success = true;
  }

  std::string pc_srv_name_;
  std::chrono::duration<double> timeout_;

  rclcpp::CallbackGroup::SharedPtr cbg_srv_;
  rclcpp::CallbackGroup::SharedPtr cbg_clients_;

  std::unordered_map<std::string, std::string> detector_services_;
  std::unordered_map<std::string, rclcpp::Client<GetBbox>::SharedPtr> detectors_;

  rclcpp::Client<GetBbox>::SharedPtr detector_client_;
  rclcpp::Client<BboxToPose>::SharedPtr pc_client_;
  rclcpp::Service<DetectByClass>::SharedPtr srv_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<DetectFusionNode>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
