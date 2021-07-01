#include <iostream>
#include "detector.h"

using namespace std;

int main()
{    
     Detector* detector = new Detector;

     detector->BuildInterpreter("/home/dev-ia/coral/pycoral/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite", 1);

     std::cout<<detector->GetVersion()<<endl;
}
