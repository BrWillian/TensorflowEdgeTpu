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

    if(_model == nullptr)
    {
        std::cerr<<"Fail to build FlatBufferModel from file: "<<_modelPath<<std::endl;
        return false;
    }

    return BuildEdgeTpuInterpreterInternal(_modelPath, num_of_threads);
}
bool Detector::BuildEdgeTpuInterpreterInternal(const char *_modelPath, const unsigned int num_of_threads)
{
    std::cout<<"Build EdgeTpu Interpreter."<<_modelPath<<std::endl;
    _edgeTpu = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();


    // Criar contexto EdgeTpu
    if(_edgeTpu == nullptr)
    {
        std::cerr<<"Fail create edge tpu context."<<std::endl;
        return false;
    }

    // Construir interpretador
    _resolver = new tflite::ops::builtin::BuiltinOpResolver();
    _resolver->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    if(tflite::InterpreterBuilder(*_model, *_resolver)(&_interpreter) != kTfLiteOk)
    {
        std::cerr<<"Fail to build interpreter."<<std::endl;
        return false;
    }


    // Inserir contexto ao interpretador
    _interpreter->SetExternalContext(kTfLiteEdgeTpuContext, _edgeTpu.get());

    if(_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr<<"Failed to allocate tensors."<<std::endl;
        return false;
    }
    std::cout<<"Success allocate tensors."<<std::endl;
}
