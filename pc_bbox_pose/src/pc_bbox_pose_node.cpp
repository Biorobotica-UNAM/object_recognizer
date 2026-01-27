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

struct CloudStamped {
  rclcpp::Time stamp;
  PointCloud2 cloud;
};

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
    CloudStamped cs;
    cs.stamp = rclcpp::Time(msg->header.stamp);
    cs.cloud = *msg;

    buffer_.push_back(std::move(cs));
    while ((int)buffer_.size() > buffer_size_) buffer_.pop_front();
  }

  const PointCloud2* pickClosest(const rclcpp::Time& t) {
    if (buffer_.empty()) return nullptr;
    auto it = std::min_element(buffer_.begin(), buffer_.end(),
      [&](const CloudStamped& a, const CloudStamped& b){
        return std::abs((a.stamp - t).nanoseconds()) < std::abs((b.stamp - t).nanoseconds());
      }
    );
    if (it == buffer_.end()) return nullptr;
    if ((it->stamp - t) > max_dt_ || (t - it->stamp) > max_dt_) return nullptr;
    return &it->cloud;
  }

  static bool getFieldOffset(const PointCloud2& cloud, const std::string& name, uint32_t &offset) {
    for (const auto &f : cloud.fields) {
      if (f.name == name) { offset = f.offset; return true; }
    }
    return false;
  }

  static inline bool readXYZat(const PointCloud2& cloud, int u, int v, float &x, float &y, float &z,
                               uint32_t ox, uint32_t oy, uint32_t oz) {
    if (cloud.height <= 1) return false;
    if (u < 0 || v < 0 || u >= (int)cloud.width || v >= (int)cloud.height) return false;

    const size_t base = (size_t)v * cloud.row_step + (size_t)u * cloud.point_step;
    std::memcpy(&x, &cloud.data[base + ox], sizeof(float));
    std::memcpy(&y, &cloud.data[base + oy], sizeof(float));
    std::memcpy(&z, &cloud.data[base + oz], sizeof(float));

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    if (z <= 0.0f) return false;
    return true;
  }

  static float median(std::vector<float> &v) {
    const size_t n = v.size();
    if (n == 0) return std::nanf("");
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    float m = v[mid];
    if (n % 2 == 0) {
      std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
      m = 0.5f * (m + v[mid - 1]);
    }
    return m;
  }

  void handle(const std::shared_ptr<BboxToPose::Request> req,
              std::shared_ptr<BboxToPose::Response> res) {
    const rclcpp::Time t(req->header.stamp);
    const PointCloud2* cloud = pickClosest(t);
    if (!cloud) { res->success = false; return; }

    if (cloud->height <= 1) { res->success = false; return; }

    uint32_t ox, oy, oz;
    if (!getFieldOffset(*cloud, "x", ox) || !getFieldOffset(*cloud, "y", oy) || !getFieldOffset(*cloud, "z", oz)) {
      res->success = false; return;
    }
    // asumiendo float32: datatype=FLOAT32 (7). Si necesitas, lo validamos luego.

    int x1 = std::max(0, std::min((int)req->x1, (int)cloud->width - 1));
    int x2 = std::max(0, std::min((int)req->x2, (int)cloud->width - 1));
    int y1 = std::max(0, std::min((int)req->y1, (int)cloud->height - 1));
    int y2 = std::max(0, std::min((int)req->y2, (int)cloud->height - 1));
    if (x2 <= x1 || y2 <= y1) { res->success = false; return; }

    std::vector<float> xs, ys, zs;
    xs.reserve(1024); ys.reserve(1024); zs.reserve(1024);

    for (int v = y1; v < y2; v += sample_step_) {
      for (int u = x1; u < x2; u += sample_step_) {
        float x, y, z;
        if (readXYZat(*cloud, u, v, x, y, z, ox, oy, oz)) {
          xs.push_back(x); ys.push_back(y); zs.push_back(z);
        }
      }
    }

    if ((int)xs.size() < min_valid_) {
      // fallback al centro
      int uc = (x1 + x2) / 2;
      int vc = (y1 + y2) / 2;
      float x, y, z;
      if (!readXYZat(*cloud, uc, vc, x, y, z, ox, oy, oz)) {
        res->success = false; return;
      }
      geometry_msgs::msg::PoseStamped pose;
      pose.header = cloud->header;
      pose.pose.position.x = x;
      pose.pose.position.y = y;
      pose.pose.position.z = z;
      pose.pose.orientation.w = 1.0;
      res->pose = pose;
      res->success = true;
      return;
    }

    float mx = median(xs);
    float my = median(ys);
    float mz = median(zs);

    if (!std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(mz)) {
      res->success = false; return;
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header = cloud->header;
    pose.pose.position.x = mx;
    pose.pose.position.y = my;
    pose.pose.position.z = mz;
    pose.pose.orientation.w = 1.0;

    res->pose = pose;
    res->success = true;
  }

  std::string cloud_topic_;
  int buffer_size_;
  std::chrono::duration<double> max_dt_;
  int sample_step_;
  int min_valid_;

  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
  rclcpp::Service<BboxToPose>::SharedPtr srv_;
  std::deque<CloudStamped> buffer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PcBboxPoseNode>());
  rclcpp::shutdown();
  return 0;
}
