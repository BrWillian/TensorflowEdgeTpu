#include <iostream>
#include "detector.h"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem::v1;

using namespace std;

int main()
{    
     Detector* detector = new Detector;

     detector->BuildInterpreter("/home/dev-ia/Downloads/model_edgetpu.tflite");

     std::string path = "/media/dev-ia/0BF3-CEB9/EQUIP80/2021/05/";

     const char *classes[] = {"carro", "moto", "onibus", "caminhão", "reboque", "caminhonete", "van"};


     for(const auto& entry: fs::directory_iterator(path))
     {
         const auto& start_time = std::chrono::steady_clock::now();
         std::chrono::duration<double, std::milli> inference_time_span;

         cv::Mat mat, input_im;
         mat = cv::imread(entry.path());

         auto width = detector->Width();
         auto height = detector->Height();

         cv::resize(mat, input_im, cv::Size(width, height));

         std::vector<uint8_t> input_data(input_im.data, input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

         const auto& result = detector->RunInference(input_data, inference_time_span);
         std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;
         std::ostringstream time_caption;

         time_caption << "Time for run inference: " << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms, " << 1000.0 / time_span.count() << "FPS";

         LOG(INFO) << time_caption.str()<<std::endl;

         std::cout<< entry.path()<<std::endl;

         cv::imshow("Image", mat);
         cv::waitKey(0);
     }
}
