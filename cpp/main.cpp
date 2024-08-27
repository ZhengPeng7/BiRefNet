#include <iostream>
#include <string>
#include <tuple>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "birefnet.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace std;

// Helper function to replace all occurrences of a character in a string
void replaceChar(std::string& str, char find, char replace) {
    size_t pos = 0;
    while ((pos = str.find(find, pos)) != std::string::npos) {
        str[pos] = replace;
        pos++;
    }
}

bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

bool createFolder(const std::string& folderPath) {
#ifdef _WIN32
    if (!CreateDirectory(folderPath.c_str(), NULL)) {
        DWORD error = GetLastError();
        if (error == ERROR_ALREADY_EXISTS) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        }
        else {
            std::cerr << "Failed to create folder! Error code: " << error << std::endl;
            return false; // Failed to create folder
        }
    }
#else
    if (mkdir(folderPath.c_str(), 0777) != 0) {
        if (errno == EEXIST) {
            std::cout << "Folder already exists!" << std::endl;
            return true; // Folder already exists
        }
        else {
            std::cerr << "Failed to create folder! Error code: " << errno << std::endl;
            return false; // Failed to create folder
        }
    }
#endif
    std::cout << "Folder created successfully!" << std::endl;
    return true; // Folder created successfully
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv)
{
    const std::string engine_file_path{ argv[1] };
    const std::string path{ argv[2] };
    std::vector<std::string> imagePathList;
    bool                     isVideo{ false };
    assert(argc == 3);

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsPathExist(path))
    {
        cv::glob(path + "/*.jpg", imagePathList);
    }
    // Assume it's a folder, add logic to handle folders
    // init model
    cout << "Loading model from " << engine_file_path << "..." << endl;
    BiRefNet birefnet_model(engine_file_path, logger);
    cout << "The model has been successfully loaded!" << endl;

    if (isVideo) {
        //path to video
        string VideoPath = path;
        // open cap
        cv::VideoCapture cap(VideoPath);

        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // Create a VideoWriter object to save the processed video
        cv::VideoWriter output_video("output_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(width, height));
        while (1)
        {
            cv::Mat frame;
            cv::Mat show_frame;
            cap >> frame;

            if (frame.empty())
                break;
            frame.copyTo(show_frame);
            cv::Mat new_frame;
            frame.copyTo(new_frame);
            auto start = std::chrono::system_clock::now();
            cv::Mat result_d = birefnet_model.predict(frame);
            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            cv::Mat result;
            cv::hconcat(show_frame, result_d, result);
            cv::resize(result, result, cv::Size(1080, 480));
            imshow("birefnet_result", result);
            output_video.write(result_d);
            cv::waitKey(0);
        }

        // Release resources
        cv::destroyAllWindows();
        cap.release();
        output_video.release();
    }
    else {
        // path to folder saves images
        string imageFolderPath_out = "results/";
        createFolder(imageFolderPath_out);
        for (const auto& imagePath : imagePathList)
        {
            // open image
            cv::Mat frame = cv::imread(imagePath);
            if (frame.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }
            cv::Mat show_frame;
            frame.copyTo(show_frame);

            auto start = chrono::system_clock::now();
            cv::Mat result_d = birefnet_model.predict(frame);
            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            //cv::Mat result;
            //cv::hconcat(show_frame, result_d, result);
            //cv::resize(result, result, cv::Size(1080, 480));
            //imshow("birefnet_result", result);
            //cv::waitKey(1);

            std::istringstream iss(imagePath);
            std::string token;
            while (std::getline(iss, token, '/'))
            {
            }
            token = token.substr(token.find_last_of("/\\") + 1);

            std::cout << "Path : " << imageFolderPath_out + token << std::endl;
            cv::imwrite(imageFolderPath_out + token, result_d);
        }
    }

    cout << "finished" << endl;
    return 0;
}
