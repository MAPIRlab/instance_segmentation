#include <instance_segmentation_utils/BufferWrapper.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <segmentation_msgs/srv/segment_image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

using Image = sensor_msgs::msg::CompressedImage;
using Detection3D = vision_msgs::msg::Detection3D;
using Detection3DArray = vision_msgs::msg::Detection3DArray;

struct CameraCalibration
{
    float centerX, centerY;
    float focalX, focalY;
    float depthLimit;
};

class ProjectTo3D : public rclcpp::Node
{
public:
    ProjectTo3D();

    void processImage();

private:
    void projectInstancesAndPublish(segmentation_msgs::srv::SegmentImage::Response::SharedPtr response, const cv::Mat& depth,
                                    std_msgs::msg::Header image_header);

    CameraCalibration m_cameraCalibration;

    Image::SharedPtr m_lastImage;
    rclcpp::Subscription<Image>::SharedPtr m_cameraSub;
    rclcpp::Client<segmentation_msgs::srv::SegmentImage>::SharedPtr m_detectronClient;
    rclcpp::Publisher<Detection3DArray>::SharedPtr m_pub3D;
    BufferWrapper m_tfBuffer;
};