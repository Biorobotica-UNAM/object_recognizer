#include <chrono>
#include <memory>
#include <string>
#include <future>

#include "rclcpp/rclcpp.hpp"
#include "detector_interfaces/srv/get_bbox.hpp"
#include "detector_interfaces/srv/bbox_to_pose.hpp"
#include "detector_interfaces/srv/detect_by_class.hpp"

using detector_interfaces::srv::GetBbox;
using detector_interfaces::srv::BboxToPose;
using detector_interfaces::srv::DetectByClass;


class DetectFusionNode
  : public rclcpp::Node
  , public std::enable_shared_from_this<DetectFusionNode>
{
public:
  DetectFusionNode() : Node("detect_fusion_node")
  {
    declare_parameter<double>("timeout_sec", 1.0);
    timeout_ = std::chrono::duration<double>(get_parameter("timeout_sec").as_double());

    detector_services_ = {
      {"YOLO", "/yolo/get_bbox"},
      {"OWL",  "/owl/get_bbox"},
      //{"DINO", "/dino/get_bbox"}
    };

    pc_srv_name_   = "/bbox_to_pose";

    // Callback groups:
    // - service callback: mutually exclusive (evita reentrancia del propio service)
    // - clients: reentrant (permite procesar respuestas mientras esperamos futures)
    cbg_srv_     = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cbg_clients_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    //yolo_client_ = this->create_client<GetBbox>(yolo_srv_name_, rclcpp::ServicesQoS(), cbg_clients_);
    for (const auto& [model, service_name] : detector_services_){
      detectors_[model] = this->create_client<GetBbox>(service_name, rclcpp::ServicesQoS(), cbg_clients_);
      RCLCPP_INFO(get_logger(), "Detector client for %s created, service: %s", model.c_str(), service_name.c_str());
    }

    pc_client_   = this->create_client<BboxToPose>(pc_srv_name_, rclcpp::ServicesQoS(), cbg_clients_);

    srv_ = this->create_service<DetectByClass>(
      "/detect_by_class",
      std::bind(&DetectFusionNode::handle, this, std::placeholders::_1, std::placeholders::_2),
      rclcpp::ServicesQoS(), cbg_srv_);

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
    auto detector_client_ = it->second;


    // Espera a que existan servicios (rápido)
    if (!detector_client_->wait_for_service(std::chrono::milliseconds(500))) {
      res->message = ("Detector %s service not available", req->model.c_str());
      return;
    }
    if (!pc_client_->wait_for_service(std::chrono::milliseconds(500))) {
      res->message = "PointCloud service not available";
      return;
    }

    auto node_base = this->get_node_base_interface();

    // 1) Call Detector service
    auto dreq = std::make_shared<GetBbox::Request>();
    dreq->class_name = req->class_name;

    auto d_future = detector_client_->async_send_request(dreq);

    if (d_future.wait_for(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_)) !=
        std::future_status::ready)
    {
      res->message = "PointCloud call failed/timeout";
      return;
    }

    auto dres = d_future.get();
    if (!dres->success) {
      res->message = "YOLO returned success=false";
      return;
    }

    // 2) Call PointCloud service
    auto preq = std::make_shared<BboxToPose::Request>();
    preq->header = dres->header;
    preq->x1 = dres->x1;
    preq->y1 = dres->y1;
    preq->x2 = dres->x2;
    preq->y2 = dres->y2;

    auto p_future = pc_client_->async_send_request(preq);

    if (p_future.wait_for(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_)) !=
        std::future_status::ready)
    {
      res->message = "PointCloud call failed/timeout";
      return;
    }

    auto pres = p_future.get();
    if (!pres->success) {
      res->message = "PointCloud returned success=false";
      return;
    }

    // 3) Respose generation
    res->pose = pres->pose;
    res->message = "Detection successful";
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

  // CRÍTICO: MultiThreadedExecutor para evitar atorón mientras esperas futures
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
