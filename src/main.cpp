#include <iostream>
#include "detector.h"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <opencv2/highgui/highgui_c.h>

namespace fs = std::experimental::filesystem::v1;

using namespace std;

int main(int argc, char *argv[])
{
     std::cout<< argv[0] << argv[1] <<std::endl;
     Detector* detector = new Detector;

     detector->BuildInterpreter(argv[1]);

     std::string path = argv[2];


     const string classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};

     for(const auto& entry: fs::directory_iterator(path))
     {
         try{
             const auto& start_time = std::chrono::steady_clock::now();
             std::chrono::duration<double, std::milli> inference_time_span;

             cv::Mat mat, input_im;
             mat = cv::imread(entry.path());

             auto width = detector->Width();
             auto height = detector->Height();

             cv::Rect roi(0, 0, 720, 405);

             mat = mat(roi);

             cv::resize(mat, input_im, cv::Size(width, height));

             std::vector<uint8_t> input_data(input_im.data, input_im.data + (input_im.cols * input_im.rows * input_im.elemSize()));

             const auto& result = detector->RunInference(input_data);


             std::cout<<"----------------------------------"<<std::endl;
             std::cout<<"Classificação: "<<entry.path().filename()<<std::endl;

             for(const auto& obj: *result)
             {
                 std::cout<<classes[obj.class_id]<<" "<<obj.confidence<<std::endl;
             }
             std::cout<<"----------------------------------"<<std::endl;


             std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;
             std::ostringstream time_caption;

             for(const auto& obj: *result)
             {
                 cv::rectangle(mat, cv::Rect(obj.x*mat.cols, obj.y*mat.rows, obj.width*mat.cols, obj.height*mat.rows),cv::Scalar(0, 255, 0));
                 cv::putText(mat, classes[obj.class_id]+" "+std::to_string(obj.confidence), cv::Point(obj.x*mat.cols, obj.y*mat.rows+10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                 cv::imwrite(entry.path(), mat);

             }


             time_caption << "Time for run inference: " << std::fixed << std::setprecision(2) << time_span.count() << " ms, " << 1000.0 / time_span.count() << "FPS";
             LOG(INFO) << time_caption.str()<<std::endl;
             LOG(INFO) << entry.path().filename()<<std::endl;

         }catch(const std::string& ex)
         {
            std::cout<<ex<<std::endl;
         }
     }
}
