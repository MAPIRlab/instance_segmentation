#pragma once
#include <image_geometry/pinhole_camera_model.h>
#include <instance_segmentation_utils/BufferWrapper.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <segmentation_msgs/srv/segment_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
using Detection3D = vision_msgs::msg::Detection3D;
using Detection3DArray = vision_msgs::msg::Detection3DArray;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;
using visualization_msgs::msg::Marker;

struct MinMaxBounds
{
    geometry_msgs::msg::Point min;
    geometry_msgs::msg::Point max;
};

enum class DepthType{Unity, Millimiters};

class ProjectTo3D : public rclcpp::Node
{
public:
    ProjectTo3D();

    void processImage();

private:
    void projectInstancesAndPublish(segmentation_msgs::srv::SegmentImage::Response::SharedPtr response, const cv::Mat& depth,
                                    const image_geometry::PinholeCameraModel& cameraModel, std_msgs::msg::Header image_header);

    void imageCallback(const Image::ConstSharedPtr& colorMsg, const Image::ConstSharedPtr& depthMsg, const CameraInfo::ConstSharedPtr& cameraInfo);
    float depthToMeters(double depth, DepthType type);
    MinMaxBounds getBoundsSingleMask(const cv::Mat& mask, const cv::Mat& coord3D, const std_msgs::msg::Header& header);
    void clearMarkers();
    void visualizeBB(const vision_msgs::msg::BoundingBox3D& box);
    void visualizePointCloud(const cv::Mat& mask, const cv::Mat& coord3D, const std_msgs::msg::Header& header);

    struct Message
    {
        Image::ConstSharedPtr colorImage;
        Image::ConstSharedPtr depthImage;
        CameraInfo::ConstSharedPtr cameraInfo;
        bool valid()
        {
            return colorImage != nullptr;
        }
    };
    Message m_lastMessage;
    message_filters::Subscriber<Image> m_colorSub;
    message_filters::Subscriber<Image> m_depthSub;
    message_filters::Subscriber<CameraInfo> m_cameraInfoSub;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer; // TODO does this need to be a shared_ptr?

    rclcpp::Client<segmentation_msgs::srv::SegmentImage>::SharedPtr m_detectronClient;
    rclcpp::Publisher<Detection3DArray>::SharedPtr m_pub3D;
    rclcpp::Publisher<Marker>::SharedPtr m_markerPub;
    rclcpp::Publisher<PointCloud2>::SharedPtr m_pclPub;
    BufferWrapper m_tfBuffer;

    std::string m_colorFormat;
    std::string m_depthFormat;
};