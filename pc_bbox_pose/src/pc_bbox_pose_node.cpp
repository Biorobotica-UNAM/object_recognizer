#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vision_interfaces/srv/bbox_to_pose.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <chrono>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using sensor_msgs::msg::PointCloud2;
using vision_interfaces::srv::BboxToPose;

class PcBboxPoseNode : public rclcpp::Node {
public:
  PcBboxPoseNode() : Node("pc_bbox_pose_node") {
    declare_parameter<std::string>("pointcloud_topic", "/hardware/camera/points");
    declare_parameter<int>("buffer_size", 15);
    declare_parameter<double>("max_dt_sec", 0.08);
    declare_parameter<int>("sample_step", 3);
    declare_parameter<int>("min_valid", 20);

    cloud_topic_ = get_parameter("pointcloud_topic").as_string();
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

   void handle(const std::shared_ptr<vision_interfaces::srv::BboxToPose::Request> req,
            std::shared_ptr<vision_interfaces::srv::BboxToPose::Response> res) {

    if (!last_cloud_) {
        RCLCPP_WARN(get_logger(), "No PointCloud received yet");
        res->success = false; return;
    }

    const auto &cloud = *last_cloud_;
    uint32_t ox, oy, oz;
    if (!getOffset(cloud, "x", ox) || !getOffset(cloud, "y", oy) || !getOffset(cloud, "z", oz)) {
        res->success = false; return;
    }

    // --- MEJORA 1: ROI CROP (Reducir 15% para evitar bordes/mesa) ---
    int w = req->x2 - req->x1;
    int h = req->y2 - req->y1;
    int x1 = std::clamp((int)(req->x1 + w * 0.15), 0, (int)cloud.width - 1);
    int x2 = std::clamp((int)(req->x2 - w * 0.15), 0, (int)cloud.width - 1);
    int y1 = std::clamp((int)(req->y1 + h * 0.15), 0, (int)cloud.height - 1);
    int y2 = std::clamp((int)(req->y2 - h * 0.15), 0, (int)cloud.height - 1);

    std::vector<Eigen::Vector3f> points;
    std::vector<float> zs;

    for (int v = y1; v < y2; v += sample_step_) {
        for (int u = x1; u < x2; u += sample_step_) {
            float x, y, z;
            if (readXYZ(cloud, u, v, x, y, z, ox, oy, oz)) {
                points.push_back(Eigen::Vector3f(x, y, z));
                zs.push_back(z);
            }
        }
    }

    if (points.size() < min_valid_) {
        res->success = false; return;
    }

    // --- MEJORA 2: FILTRO DE PROFUNDIDAD (Z-Score simple) ---
    float med_z = median(zs);
    std::vector<Eigen::Vector3f> inliers;
    for (const auto& p : points) {
        if (std::abs(p.z() - med_z) < 0.08) { // Solo puntos a +/- 8cm de la mediana
            inliers.push_back(p);
        }
    }

    if (inliers.size() < 10) { res->success = false; return; }

    // --- MEJORA 3: PCA ALINEADO PARA GRASPING ---
    Eigen::Vector3f centroid(0, 0, 0);
    for (const auto& p : inliers) centroid += p;
    centroid /= inliers.size();

    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto& p : inliers) {
        Eigen::Vector3f centered = p - centroid;
        cov += centered * centered.transpose();
    }
    cov /= inliers.size() - 1;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    // Eje Z (Normal al objeto): Eigenvector con menor varianza
    Eigen::Vector3f z_axis = es.eigenvectors().col(0); 
    // Eje X (Largo del objeto): Eigenvector con mayor varianza
    Eigen::Vector3f x_axis = es.eigenvectors().col(2); 

    // Asegurar que el eje Z apunte hacia la cámara para que el gripper no llegue por "atrás"
    if (z_axis.z() > 0) z_axis = -z_axis;
    
    // Ortogonalizar
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    x_axis = y_axis.cross(z_axis); 

    Eigen::Matrix3f rot;
    rot << x_axis, y_axis, z_axis;
    Eigen::Quaternionf q(rot);
    q.normalize();

    res->pose.header = cloud.header;
    res->pose.pose.position.x = centroid.x();
    res->pose.pose.position.y = centroid.y();
    res->pose.pose.position.z = centroid.z();
    res->pose.pose.orientation.x = q.x();
    res->pose.pose.orientation.y = q.y();
    res->pose.pose.orientation.z = q.z();
    res->pose.pose.orientation.w = q.w();
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
  rclcpp::Service<vision_interfaces::srv::BboxToPose>::SharedPtr srv_;

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
