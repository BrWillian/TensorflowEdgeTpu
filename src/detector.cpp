#include "detector.h"

Detector::Detector()
{

}
const char* Detector::GetVersion()
{
    return "1.0.0";
}
bool Detector::BuildInterpreter(const char *_modelPath, const unsigned int num_of_threads)
{
    _model = tflite::FlatBufferModel::BuildFromFile(_modelPath);


    return true;
}
