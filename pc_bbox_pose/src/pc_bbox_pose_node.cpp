#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <vision_interfaces/srv/bbox_to_pose.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <mutex>
#include <deque>
#include <limits>
#include <stdexcept>

using sensor_msgs::msg::PointCloud2;
using vision_interfaces::srv::BboxToPose;

class PcBboxPoseNode : public rclcpp::Node
{
public:

    PcBboxPoseNode()
        : Node("pc_bbox_pose_node")
    {
        declare_parameter<std::string>(
            "pointcloud_topic",
            "/hardware/camera/points");

        declare_parameter<std::string>(
            "target_frame",
            "base_link");

        declare_parameter<int>("sample_step", 1);
        declare_parameter<int>("min_valid", 20);
        declare_parameter<double>("depth_offset_m", 0.0);
        declare_parameter<std::string>("depth_axis", "auto");

        cloud_topic_ =
            get_parameter("pointcloud_topic").as_string();

        target_frame_ =
            get_parameter("target_frame").as_string();

        sample_step_ =
            get_parameter("sample_step").as_int();

        min_valid_ =
            get_parameter("min_valid").as_int();

        depth_offset_ =
            static_cast<float>(
                get_parameter("depth_offset_m").as_double());

        depth_axis_ =
            get_parameter("depth_axis").as_string();

        if (depth_axis_ != "x" &&
            depth_axis_ != "y" &&
            depth_axis_ != "z" &&
            depth_axis_ != "auto")
        {
            RCLCPP_WARN(
                get_logger(),
                "Invalid depth_axis='%s'. Falling back to 'auto'",
                depth_axis_.c_str());

            depth_axis_ = "auto";
        }

        declare_parameter<int>("cloud_history_size", 60);
        declare_parameter<double>("max_sync_delta_sec", 0.10);
        cloud_history_size_ = get_parameter("cloud_history_size").as_int();
        max_sync_delta_sec_ = get_parameter("max_sync_delta_sec").as_double();
        if (cloud_history_size_ < 1 || !std::isfinite(max_sync_delta_sec_) ||
            max_sync_delta_sec_ < 0.0 || sample_step_ < 1 || min_valid_ < 1)
        {
            throw std::invalid_argument("Invalid cloud history, sync tolerance or sampling parameters");
        }

        sub_ =
            create_subscription<PointCloud2>(
                cloud_topic_,
                rclcpp::SensorDataQoS(),
                std::bind(
                    &PcBboxPoseNode::cloudCb,
                    this,
                    std::placeholders::_1));

        srv_ =
            create_service<BboxToPose>(
                "/vision/bbox_to_pose",
                std::bind(
                    &PcBboxPoseNode::handle,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2));

        tf_buffer_ =
            std::make_shared<tf2_ros::Buffer>(
                get_clock());

        tf_listener_ =
            std::make_shared<tf2_ros::TransformListener>(
                *tf_buffer_);

        RCLCPP_INFO(
            get_logger(),
            "PC bbox->pose ready. topic=%s target_frame=%s depth_axis=%s",
            cloud_topic_.c_str(),
            target_frame_.c_str(),
            depth_axis_.c_str());
    }

private:

    void cloudCb(const PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lk(cloud_mutex_);
        const auto stamp = stampNanoseconds(msg->header.stamp);
        if (stamp <= 0)
        {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Ignoring PointCloud without valid timestamp");
            return;
        }
        // Clear history on clock reset (e.g. a simulation restart).
        if (!cloud_history_.empty() &&
            stamp < stampNanoseconds(cloud_history_.back()->header.stamp))
        {
            cloud_history_.clear();
        }
        cloud_history_.push_back(msg);
        while (cloud_history_.size() > static_cast<size_t>(cloud_history_size_))
        {
            cloud_history_.pop_front();
        }
    }

    void handle(
        const std::shared_ptr<BboxToPose::Request> req,
        std::shared_ptr<BboxToPose::Response> res)
    {
        res->success = false;
        const auto requested_stamp = stampNanoseconds(req->header.stamp);
        if (requested_stamp <= 0)
        {
            RCLCPP_WARN(get_logger(), "BboxToPose requires the source image timestamp");
            return;
        }
        PointCloud2::SharedPtr cloud_snap;
        int64_t best_delta = std::numeric_limits<int64_t>::max();
        {
            std::lock_guard<std::mutex> lk(cloud_mutex_);
            for (const auto& candidate : cloud_history_)
            {
                const auto stamp = stampNanoseconds(candidate->header.stamp);
                const auto delta = stamp >= requested_stamp ? stamp - requested_stamp
                                                          : requested_stamp - stamp;
                if (delta < best_delta)
                {
                    best_delta = delta;
                    cloud_snap = candidate;
                }
            }
        }
        if (!cloud_snap || static_cast<double>(best_delta) / 1e9 > max_sync_delta_sec_)
        {
            RCLCPP_WARN(get_logger(),
                        "No PointCloud matching image timestamp (nearest delta=%.3f s, limit=%.3f s)",
                        cloud_snap ? static_cast<double>(best_delta) / 1e9 : -1.0,
                        max_sync_delta_sec_);
            return;
        }

        const auto& cloud = *cloud_snap;
        // Pixel bboxes require an organized cloud registered to the RGB image.
        if (cloud.width == 0 || cloud.height <= 1 || req->x1 < 0 || req->y1 < 0 ||
            req->x2 <= req->x1 || req->y2 <= req->y1 ||
            static_cast<uint32_t>(req->x2) > cloud.width ||
            static_cast<uint32_t>(req->y2) > cloud.height)
        {
            RCLCPP_WARN(get_logger(), "Invalid bbox or incompatible organized cloud dimensions");
            return;
        }
        if (cloud.is_bigendian || cloud.point_step == 0 ||
            static_cast<uint64_t>(cloud.row_step) <
                static_cast<uint64_t>(cloud.width) * cloud.point_step ||
            cloud.data.size() < static_cast<uint64_t>(cloud.row_step) * cloud.height)
        {
            RCLCPP_WARN(get_logger(), "Unsupported or incomplete PointCloud layout");
            return;
        }
        RCLCPP_DEBUG(get_logger(), "Selected PointCloud with image delta=%.3f s",
                     static_cast<double>(best_delta) / 1e9);

        uint32_t ox, oy, oz;

        if (!getOffset(cloud, "x", ox) ||
            !getOffset(cloud, "y", oy) ||
            !getOffset(cloud, "z", oz))
        {
            RCLCPP_ERROR(
                get_logger(),
                "PointCloud missing XYZ fields");

            res->success = false;
            return;
        }

        // ROI crop
        int w = req->x2 - req->x1;
        int h = req->y2 - req->y1;

        int x1 =
            std::clamp(
                static_cast<int>(req->x1 + w * 0.15),
                0,
                static_cast<int>(cloud.width) - 1);

        int x2 =
            std::clamp(
                static_cast<int>(req->x2 - w * 0.15),
                0,
                static_cast<int>(cloud.width) - 1);

        int y1 =
            std::clamp(
                static_cast<int>(req->y1 + h * 0.15),
                0,
                static_cast<int>(cloud.height) - 1);

        int y2 =
            std::clamp(
                static_cast<int>(req->y2 - h * 0.15),
                0,
                static_cast<int>(cloud.height) - 1);

        std::vector<Eigen::Vector3f> points;

        for (int v = y1; v < y2; v += sample_step_)
        {
            for (int u = x1; u < x2; u += sample_step_)
            {
                float x, y, z;

                if (readXYZ(
                        cloud,
                        u,
                        v,
                        x,
                        y,
                        z,
                        ox,
                        oy,
                        oz))
                {
                    points.emplace_back(x, y, z);
                }
            }
        }

        if (points.size() < static_cast<size_t>(min_valid_))
        {
            RCLCPP_WARN(
                get_logger(),
                "Not enough valid points");

            res->success = false;
            return;
        }

        const std::string effective_depth_axis =
            effectiveDepthAxis(cloud.header.frame_id);

        std::vector<float> depth_samples;
        depth_samples.reserve(points.size());
        for (const auto& p : points)
        {
            depth_samples.push_back(depthValue(p, effective_depth_axis));
        }

        RCLCPP_INFO(
            get_logger(),
            "BboxToPose cloud_frame=%s depth_axis=%s",
            cloud.header.frame_id.c_str(),
            effective_depth_axis.c_str());

        // Depth filter
        float med_depth = median(depth_samples);

        std::vector<Eigen::Vector3f> inliers;

        for (const auto& p : points)
        {
            if (std::abs(depthValue(p, effective_depth_axis) - med_depth) < 0.15f)
            {
                inliers.push_back(p);
            }
        }

        if (inliers.size() < 10)
        {
            RCLCPP_WARN(
                get_logger(),
                "Not enough inliers");

            res->success = false;
            return;
        }

        // Centroid
        Eigen::Vector3f centroid(0.f, 0.f, 0.f);

        for (const auto& p : inliers)
        {
            centroid += p;
        }

        centroid /= static_cast<float>(inliers.size());

        // PCA

        float bbox_h = static_cast<float>(req->y2 - req->y1);
        float bbox_w = static_cast<float>(req->x2 - req->x1);
        bool is_vertical = (bbox_h / (bbox_w + 1e-3f)) > 1.4f;

        RCLCPP_INFO(
            get_logger(),
            "bbox aspect h/w=%.2f -> %s",
            bbox_h / (bbox_w + 1e-3f),
            is_vertical ? "VERTICAL" : "HORIZONTAL"
        );

        applyDepthOffset(centroid, effective_depth_axis);
        //Eigen::Matrix3f cov =
        //    Eigen::Matrix3f::Zero();

        //for (const auto& p : inliers)
        //{
        //    Eigen::Vector3f centered =
        //        p - centroid;

        //    cov +=
        //        centered *
        //        centered.transpose();
        //}

        //cov /= static_cast<float>(inliers.size() - 1);

        //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);

        //Eigen::Vector3f major_axis =
        //    es.eigenvectors().col(2);

        //float vertical_component =
        //    std::abs(
        //        major_axis.dot(
        //            Eigen::Vector3f::UnitY()));

        //bool is_vertical =
        //    vertical_component > 0.707f;

        geometry_msgs::msg::PoseStamped pose_camera;

        pose_camera.header =
            cloud.header;

        pose_camera.pose.position.x =
            centroid.x();

        pose_camera.pose.position.y =
            centroid.y();

        pose_camera.pose.position.z =
            centroid.z();

        if (is_vertical)
        {
            RCLCPP_INFO(
                get_logger(),
                "VERTICAL OBJECT -> SIDE grasp");

            pose_camera.pose.orientation.x = 0.0;
            pose_camera.pose.orientation.y = -0.7071068;
            pose_camera.pose.orientation.z = 0.0;
            pose_camera.pose.orientation.w = 0.7071068;
        }
        else
        {
            RCLCPP_INFO(
                get_logger(),
                "HORIZONTAL OBJECT -> TOP grasp");

            pose_camera.pose.orientation.x = 0.0;
            pose_camera.pose.orientation.y = 0.0;
            pose_camera.pose.orientation.z = 0.0;
            pose_camera.pose.orientation.w = 1.0;
        }

        geometry_msgs::msg::PoseStamped pose_target;

        try
        {
            geometry_msgs::msg::PointStamped point_camera;
			point_camera.header = pose_camera.header;
			point_camera.point.x = centroid.x();
			point_camera.point.y = centroid.y();
			point_camera.point.z = centroid.z();

			geometry_msgs::msg::PointStamped point_target;

			point_target =
				tf_buffer_->transform(
					point_camera,
					target_frame_,
					tf2::durationFromSec(0.2));

			pose_target.header = point_target.header;

			pose_target.pose.position.x = point_target.point.x;
			pose_target.pose.position.y = point_target.point.y;
			pose_target.pose.position.z = point_target.point.z;

			if (is_vertical)
			{
				pose_target.pose.orientation.x = 0.0;
				pose_target.pose.orientation.y = -0.7071068;
				pose_target.pose.orientation.z = 0.0;
				pose_target.pose.orientation.w = 0.7071068;
			}
			else
			{
				pose_target.pose.orientation.x = 0.0;
				pose_target.pose.orientation.y = -0.3826834;
				pose_target.pose.orientation.z = 0.0;
				pose_target.pose.orientation.w = 0.9238795;
			}

            res->pose =
                pose_target;

            res->success = true;

            RCLCPP_INFO(
                get_logger(),
                "Pose transformed %s -> %s",
                pose_camera.header.frame_id.c_str(),
                target_frame_.c_str());

            RCLCPP_INFO(
                get_logger(),
                "Final pose: "
                "pos=[%.3f %.3f %.3f] "
                "quat=[%.3f %.3f %.3f %.3f]",
                res->pose.pose.position.x,
                res->pose.pose.position.y,
                res->pose.pose.position.z,
                res->pose.pose.orientation.x,
                res->pose.pose.orientation.y,
                res->pose.pose.orientation.z,
                res->pose.pose.orientation.w);
        }
        catch (const tf2::TransformException& ex)
        {
            RCLCPP_ERROR(
                get_logger(),
                "TF transform failed: %s",
                ex.what());

            res->success = false;
        }
    }

    static bool getOffset(
        const PointCloud2& cloud,
        const std::string& name,
        uint32_t& offset)
    {
        for (const auto& f : cloud.fields)
        {
            if (f.name == name && f.datatype == sensor_msgs::msg::PointField::FLOAT32 &&
                f.count >= 1 && static_cast<uint64_t>(f.offset) + sizeof(float) <= cloud.point_step)
            {
                offset = f.offset;
                return true;
            }
        }

        return false;
    }

    static bool readXYZ(
        const PointCloud2& cloud,
        int u,
        int v,
        float& x,
        float& y,
        float& z,
        uint32_t ox,
        uint32_t oy,
        uint32_t oz)
    {
        const size_t base =
            v * cloud.row_step +
            u * cloud.point_step;

        std::memcpy(
            &x,
            &cloud.data[base + ox],
            sizeof(float));

        std::memcpy(
            &y,
            &cloud.data[base + oy],
            sizeof(float));

        std::memcpy(
            &z,
            &cloud.data[base + oz],
            sizeof(float));

        return
            std::isfinite(x) &&
            std::isfinite(y) &&
            std::isfinite(z);
    }

    static float median(std::vector<float> v)
    {
        std::nth_element(
            v.begin(),
            v.begin() + v.size() / 2,
            v.end());

        return v[v.size() / 2];
    }

    std::string effectiveDepthAxis(const std::string& frame_id) const
    {
        if (depth_axis_ != "auto")
        {
            return depth_axis_;
        }

        if (frame_id.find("optical") != std::string::npos)
        {
            return "z";
        }

        return "x";
    }

    float depthValue(
        const Eigen::Vector3f& p,
        const std::string& axis) const
    {
        if (axis == "y")
        {
            return p.y();
        }

        if (axis == "z")
        {
            return p.z();
        }

        return p.x();
    }

    void applyDepthOffset(
        Eigen::Vector3f& p,
        const std::string& axis) const
    {
        if (axis == "y")
        {
            p.y() += depth_offset_;
            return;
        }

        if (axis == "z")
        {
            p.z() += depth_offset_;
            return;
        }

        p.x() += depth_offset_;
    }

    static int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp)
    {
        return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
    }

    int cloud_history_size_;
    double max_sync_delta_sec_;
    std::mutex cloud_mutex_;

    std::deque<PointCloud2::SharedPtr> cloud_history_;

    rclcpp::Subscription<PointCloud2>::SharedPtr sub_;

    rclcpp::Service<BboxToPose>::SharedPtr srv_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::string cloud_topic_;

    std::string target_frame_;

    int sample_step_;

    int min_valid_;

    float depth_offset_;

    std::string depth_axis_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<PcBboxPoseNode>());

    rclcpp::shutdown();

    return 0;
}
