#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <detector_interfaces/srv/bbox_to_pose.hpp>

#include <chrono>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using sensor_msgs::msg::PointCloud2;
using detector_interfaces::srv::BboxToPose;

class PcBboxPoseNode : public rclcpp::Node {
public:
  PcBboxPoseNode() : Node("pc_bbox_pose_node") {
    declare_parameter<std::string>("cloud_topic", "/points");
    declare_parameter<int>("buffer_size", 15);
    declare_parameter<double>("max_dt_sec", 0.08);
    declare_parameter<int>("sample_step", 3);
    declare_parameter<int>("min_valid", 20);

    cloud_topic_ = get_parameter("cloud_topic").as_string();
    buffer_size_ = get_parameter("buffer_size").as_int();
    max_dt_ = std::chrono::duration<double>(get_parameter("max_dt_sec").as_double());
    sample_step_ = get_parameter("sample_step").as_int();
    min_valid_ = get_parameter("min_valid").as_int();

    sub_ = create_subscription<PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PcBboxPoseNode::cloudCb, this, std::placeholders::_1)
    );

    srv_ = create_service<BboxToPose>(
      "bbox_to_pose",
      std::bind(&PcBboxPoseNode::handle, this, std::placeholders::_1, std::placeholders::_2)
    );

    RCLCPP_INFO(get_logger(), "PC bbox->pose ready. topic=%s", cloud_topic_.c_str());
  }

private:
  void cloudCb(const PointCloud2::SharedPtr msg) {
    last_cloud_ = msg;
  }

   void handle(const std::shared_ptr<BboxToPose::Request> req,
              std::shared_ptr<BboxToPose::Response> res) {

    if (!last_cloud_) {
      RCLCPP_WARN(get_logger(), "No PointCloud received yet");
      res->success = false;
      return;
    }

    const auto &cloud = *last_cloud_;
    if (cloud.height <= 1) {
      RCLCPP_WARN(get_logger(), "PointCloud is not organized");
      res->success = false;
      return;
    }

    uint32_t ox, oy, oz;
    if (!getOffset(cloud, "x", ox) ||
        !getOffset(cloud, "y", oy) ||
        !getOffset(cloud, "z", oz)) {
      res->success = false;
      RCLCPP_WARN(get_logger(), "PointCloud is missing x,y,or z fields");
      return;
    }

    int x1 = std::clamp((int)req->x1, 0, (int)cloud.width - 1);
    int x2 = std::clamp((int)req->x2, 0, (int)cloud.width - 1);
    int y1 = std::clamp((int)req->y1, 0, (int)cloud.height - 1);
    int y2 = std::clamp((int)req->y2, 0, (int)cloud.height - 1);

    std::vector<float> xs, ys, zs;

    for (int v = y1; v < y2; v += 3) {
      for (int u = x1; u < x2; u += 3) {
        float x, y, z;
        if (readXYZ(cloud, u, v, x, y, z, ox, oy, oz)) {
          xs.push_back(x); ys.push_back(y); zs.push_back(z);
        }
      }
    }

    if (xs.size() < 30) {
      res->success = false;
      RCLCPP_WARN(get_logger(), "PointCloud is not enough valid points (%zu)", xs.size());
      return;
    }

    auto mx = median(xs);
    auto my = median(ys);
    auto mz = median(zs);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = cloud.header;
    pose.pose.position.x = mx;
    pose.pose.position.y = my;
    pose.pose.position.z = mz;
    pose.pose.orientation.w = 1.0;

    res->pose = pose;
    res->success = true;
  }

  static bool getOffset(const PointCloud2& cloud,
                        const std::string& name, uint32_t &offset) {
    for (const auto &f : cloud.fields)
      if (f.name == name) { offset = f.offset; return true; }
    return false;
  }

  static bool readXYZ(const PointCloud2& cloud, int u, int v,
                      float &x, float &y, float &z,
                      uint32_t ox, uint32_t oy, uint32_t oz) {
    const size_t base = v * cloud.row_step + u * cloud.point_step;
    std::memcpy(&x, &cloud.data[base + ox], sizeof(float));
    std::memcpy(&y, &cloud.data[base + oy], sizeof(float));
    std::memcpy(&z, &cloud.data[base + oz], sizeof(float));
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  static float median(std::vector<float>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  }

  PointCloud2::SharedPtr last_cloud_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  rclcpp::Service<BboxToPose>::SharedPtr srv_;

  std::string cloud_topic_ ;
  int buffer_size_ ;
  std::chrono::duration<double> max_dt_ ;
  int sample_step_;
  int min_valid_ ;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PcBboxPoseNode>());
  rclcpp::shutdown();
  return 0;
}
