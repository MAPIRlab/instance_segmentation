#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <instance_segmentation_utils/projectTo3D.hpp>
#include <opencv2/highgui.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ProjectTo3D>();

    // we have a polling loop rather than just spinning and letting the callback handle everything to avoid the headache
    // of calling a service inside a callback
    // technically doable, I think, but recursive spinning is not allowed, so not trivial
    rclcpp::Rate rate(30);
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);
        node->processImage();
        rate.sleep();
    }
    return 0;
}

ProjectTo3D::ProjectTo3D() : Node("ProjectInstancesTo3D"), m_tfBuffer(get_clock())
{
    using namespace std::placeholders;

    std::string color_topic = declare_parameter<std::string>("color_topic", "camera/color/raw");
    std::string depth_topic = declare_parameter<std::string>("depth_topic", "camera/depth/raw");
    std::string camerainfo_topic = declare_parameter<std::string>("info_topic", "camera/info");
    m_colorSub.subscribe(this, color_topic);
    m_depthSub.subscribe(this, depth_topic);
    m_cameraInfoSub.subscribe(this, camerainfo_topic);
    synchronizer = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(5), m_colorSub, m_depthSub, m_cameraInfoSub);
    synchronizer->registerCallback(std::bind(&ProjectTo3D::imageCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    RCLCPP_INFO(get_logger(),
                "Using topics:\n"
                "   color: %s\n"
                "   depth: %s\n"
                "   info: %s",
                color_topic.c_str(), depth_topic.c_str(), camerainfo_topic.c_str());

    m_colorFormat = declare_parameter<std::string>("color_format", "rgb8");
    m_depthFormat = declare_parameter<std::string>("depth_format", "rgb8");

    const char* detectronTopic = "/detectron/segment";
    m_detectronClient = create_client<segmentation_msgs::srv::SegmentImage>(detectronTopic);
    m_pub3D = create_publisher<Detection3DArray>("semantic_instances_3D", 10);
    m_markerPub = create_publisher<Marker>("/semanticObject_visualization", 10);
    m_pclPub = create_publisher<PointCloud2>("/semanticPointCloud", 10);

    RCLCPP_INFO(get_logger(), "Expecting detectron service in: %s", detectronTopic);
    while (!m_detectronClient->wait_for_service(std::chrono::seconds(5))) RCLCPP_INFO(get_logger(), "Waiting for service to become available...");
    RCLCPP_INFO(get_logger(), "Found detectron, ready for action!");
}

void ProjectTo3D::processImage()
{
    if (!m_lastMessage.valid())
        return;

    RCLCPP_INFO(get_logger(), "Processing one image pair");

    // copy the info, decompressing images if necessary
    std_msgs::msg::Header header = m_lastMessage.colorImage->header;
    cv::Mat rgb = cv_bridge::toCvCopy(m_lastMessage.colorImage, m_colorFormat)->image;

    cv::Mat depth;
    // For some reason cv_bridge does not correctly parse mono16 images. Instead of a single 16 bit channel, it creates a cv::Mat with 2 8 bit
    // channels
    if (m_depthFormat == "mono16")
    {
        uint8_t* depthBytes = (uint8_t*)m_lastMessage.depthImage->data.data();
        depth = cv::Mat((int)m_lastMessage.depthImage->height, (int)m_lastMessage.depthImage->width, CV_16U, depthBytes).clone();
    }
    else
        depth = cv_bridge::toCvCopy(m_lastMessage.depthImage, m_depthFormat)->image;

    image_geometry::PinholeCameraModel cameraModel;
    cameraModel.fromCameraInfo(*m_lastMessage.cameraInfo);

    // clean up the buffer before spinning
    m_lastMessage.colorImage = nullptr;
    m_lastMessage.depthImage = nullptr;
    m_lastMessage.cameraInfo = nullptr;

    // send the rgb image to detectron
    auto request = std::make_shared<segmentation_msgs::srv::SegmentImage::Request>();
    cv_bridge::CvImage(header, m_colorFormat, rgb).toImageMsg(request->image);

    auto future = m_detectronClient->async_send_request(request);
    try
    {
        if (rclcpp::spin_until_future_complete(shared_from_this(), future) == rclcpp::FutureReturnCode::SUCCESS)
            projectInstancesAndPublish(future.get(), depth, cameraModel, header);
        else
            RCLCPP_ERROR(get_logger(), "Could not access the detectron service!");
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR(get_logger(), "Exception while processing image: %s", e.what());
    }
}

void ProjectTo3D::projectInstancesAndPublish(segmentation_msgs::srv::SegmentImage::Response::SharedPtr response, const cv::Mat& depth,
                                             const image_geometry::PinholeCameraModel& cameraModel, std_msgs::msg::Header image_header)
{
    clearMarkers();

    // 3D coordinates of the point that corresponds to each specific pixel in the image (camera space)
    cv::Mat coord3D(depth.rows, depth.cols, CV_32FC3);
    {
#pragma omp parallel for collapse(2)
        for (int column = 0; column < coord3D.cols; column++)
        {
            for (int row = 0; row < coord3D.rows; row++)
            {
                // TODO make the type be inferred from m_depthFormat
                float z = depthToMeters(depth.at<uint16_t>(row, column), DepthType::Unity);

                cv::Point2d imagePoint = cameraModel.rectifyPoint(cv::Point2d(column, row));
                cv::Point3d rayDirection = cameraModel.projectPixelTo3dRay(imagePoint);
                cv::Point3d point3D = rayDirection * z;

                coord3D.at<cv::Vec3f>(row, column) = cv::Vec3f(point3D.x, point3D.y, point3D.z);
            }
        }
    }

    // process the instances
    Detection3DArray objectsArray;
    for (const segmentation_msgs::msg::SemanticInstance2D& instance : response->instances)
    {
        objectsArray.detections.emplace_back();
        Detection3D& detection = objectsArray.detections.back();

        detection.results = instance.detection.results;

        cv::Mat mask = cv_bridge::toCvCopy(instance.mask)->image;
        // cv::imshow("mask", mask->image);
        // cv::waitKey(30);

        cv::erode(mask, mask, cv::Mat::ones(10,10, CV_8U));

        // Get the masked point cloud, and record the min and max coords in world
        MinMaxBounds bounds = getBoundsSingleMask(mask, coord3D, image_header);
        visualizePointCloud(mask, coord3D, image_header);

        detection.header.frame_id = "map";
        detection.header.stamp = image_header.stamp;

        detection.bbox.size.x = bounds.max.x - bounds.min.x;
        detection.bbox.size.y = bounds.max.y - bounds.min.y;
        detection.bbox.size.z = bounds.max.z - bounds.min.z;

        detection.bbox.center.position.x = bounds.min.x + detection.bbox.size.x * 0.5;
        detection.bbox.center.position.y = bounds.min.y + detection.bbox.size.y * 0.5;
        detection.bbox.center.position.z = bounds.min.z + detection.bbox.size.z * 0.5;
        visualizeBB(detection.bbox);
    }

    m_pub3D->publish(objectsArray);
}

MinMaxBounds ProjectTo3D::getBoundsSingleMask(const cv::Mat& mask, const cv::Mat& coord3D, const std_msgs::msg::Header& header)
{
    geometry_msgs::msg::TransformStamped cameraToMap = m_tfBuffer.buffer.lookupTransform("map", header.frame_id, tf2_ros::fromMsg(header.stamp));

    geometry_msgs::msg::Point min;
    min.x = DBL_MAX;
    min.y = DBL_MAX;
    min.z = DBL_MAX;
    geometry_msgs::msg::Point max;
    max.x = -DBL_MAX;
    max.y = -DBL_MAX;
    max.z = -DBL_MAX;
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i, j) != 0)
            {
                geometry_msgs::msg::Point cameraPoint;
                cameraPoint.x = coord3D.at<cv::Vec3f>(i,j)[0];
                cameraPoint.y = coord3D.at<cv::Vec3f>(i,j)[1];
                cameraPoint.z = coord3D.at<cv::Vec3f>(i,j)[2];

                // transform point cloud to world space to get the AABB
                geometry_msgs::msg::Point worldPoint;
                tf2::doTransform(cameraPoint, worldPoint, cameraToMap);
                min.x = std::min(min.x, worldPoint.x);
                min.y = std::min(min.y, worldPoint.y);
                min.z = std::min(min.z, worldPoint.z);

                max.x = std::max(max.x, worldPoint.x);
                max.y = std::max(max.y, worldPoint.y);
                max.z = std::max(max.z, worldPoint.z);
            }
        }
    }
    return {min, max};
}

void ProjectTo3D::imageCallback(const Image::ConstSharedPtr& colorMsg, const Image::ConstSharedPtr& depthMsg,
                                const CameraInfo::ConstSharedPtr& cameraInfo)
{
    m_lastMessage.colorImage = colorMsg;
    m_lastMessage.depthImage = depthMsg;
    m_lastMessage.cameraInfo = cameraInfo;
}

float ProjectTo3D::depthToMeters(double depth, DepthType type)
{
    if (type == DepthType::Unity)
    {
        constexpr float max_range = 10.0f;
        return (depth / std::numeric_limits<uint16_t>::max()) * max_range;
    }
    else if (type == DepthType::Millimiters)
    {
        return depth * 0.001;
    }
    else
    {
        RCLCPP_ERROR(get_logger(), "DepthType %d not supported", (int)type);
        throw std::exception();
    }
}

void ProjectTo3D::clearMarkers()
{
    Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = now();
    marker.action = Marker::DELETEALL;
    m_markerPub->publish(marker);
}

void ProjectTo3D::visualizeBB(const vision_msgs::msg::BoundingBox3D& box)
{
    Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = now();
    marker.action = Marker::ADD;
    marker.type = Marker::CUBE;
    marker.pose = box.center;
    marker.scale = box.size;

    marker.color.r = 1;
    marker.color.g = 1;
    marker.color.b = 1;
    marker.color.a = 1;

    m_markerPub->publish(marker);
}

void ProjectTo3D::visualizePointCloud(const cv::Mat& mask, const cv::Mat& coord3D, const std_msgs::msg::Header& header)
{
    pcl::PointCloud<pcl::PointXYZ> pcl;
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            if (mask.at<uint8_t>(i, j) != 0)
            {
                geometry_msgs::msg::Point cameraPoint;
                cameraPoint.x = coord3D.at<cv::Vec3f>(i,j)[0];
                cameraPoint.y = coord3D.at<cv::Vec3f>(i,j)[1];
                cameraPoint.z = coord3D.at<cv::Vec3f>(i,j)[2];
                pcl::PointXYZ pcl_point(cameraPoint.x, cameraPoint.y, cameraPoint.z);
                pcl.points.push_back(pcl_point);
            }
        }
    }
    PointCloud2 pointMsg;
    pcl::toROSMsg(pcl, pointMsg);
    pointMsg.header = header;
    m_pclPub->publish(pointMsg);
}
