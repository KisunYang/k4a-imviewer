#include <iostream>
#include <k4a/k4a.h>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    try
    {
        const uint32_t devices = k4a::device::get_installed_count();
        std::cout << "Azure Kinect detected devices: " << devices << std::endl;
        if (devices == 0)
        {
            std::cerr << "No Kinect device reported by SDK. Check USB 3.0 and cable.\n";
            return 1;
        }

        // Open default device
        k4a::device device = k4a::device::open(0);

        // Configure camera
        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        config.color_resolution = K4A_COLOR_RESOLUTION_720P;
        config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        config.camera_fps = K4A_FRAMES_PER_SECOND_30;
        config.synchronized_images_only = true;

        device.start_cameras(&config);

        cv::namedWindow("Azure Kinect Color", cv::WINDOW_NORMAL);

        while (true)
        {
            k4a::capture capture;
            if (device.get_capture(&capture, std::chrono::milliseconds(1000)))
            {
                // Get color image
                k4a::image color_image = capture.get_color_image();
                if (color_image)
                {
                    int w = color_image.get_width_pixels();
                    int h = color_image.get_height_pixels();

                    // BGRA image
                    cv::Mat color_mat(h, w, CV_8UC4, (void*)color_image.get_buffer());
                    cv::imshow("Azure Kinect Color", color_mat);

                    color_image.reset();
                }
                capture.reset();
            }

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') // ESC or Q
                break;
            if (cv::getWindowProperty("Azure Kinect Color", cv::WND_PROP_VISIBLE) < 1)
                break;
        }

        // Must stop streaming before close, or shutdown can hang.
        device.stop_cameras();
        cv::destroyAllWindows();
        device.close();

        return 0;
    }
    catch (const k4a::error& e)
    {
        std::cerr << "K4A error: " << e.what() << std::endl;
        std::cerr << "If the device is in use, close k4aviewer / other Kinect apps and retry.\n";
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
