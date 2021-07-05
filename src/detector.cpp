#include "detector.h"

Detector::Detector()
{

}
const char* Detector::GetVersion()
{
    return "1.0.0";
}
void Detector::BuildInterpreter(const char *_modelPath)
{
    _model = tflite::FlatBufferModel::BuildFromFile(_modelPath);

    if(_model == nullptr)
    {
        LOG(ERROR)<<"Fail to build FlatBufferModel from file: "<<_modelPath<<std::endl;
        exit(-1);
    }

    BuildEdgeTpuInterpreter(_modelPath);
}
void Detector::BuildEdgeTpuInterpreter(const char *_modelPath)
{
    const auto& start_time = std::chrono::steady_clock::now();

    LOG(INFO)<<"Build EdgeTpu Interpreter."<<_modelPath<<std::endl;
    _edgeTpu = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Criar contexto EdgeTpu
    if(_edgeTpu == nullptr)
    {
        LOG(ERROR)<<"Fail create edge tpu context."<<std::endl;
        exit(-1);
    }

    // Construir interpretador
    _resolver = new tflite::ops::builtin::BuiltinOpResolver();
    _resolver->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    if(tflite::InterpreterBuilder(*_model, *_resolver)(&_interpreter) != kTfLiteOk)
    {
        LOG(ERROR)<<"Fail to build interpreter."<<std::endl;
        exit(-1);
    }

    // Verificar tipo de entrada do tensor

    int inputTmp = _interpreter->inputs()[0];
    if(_interpreter->tensor(inputTmp)->type == kTfLiteFloat32)
    {
        LOG(INFO)<<"Floating point Tensorflow lite model"<<std::endl;
    }

    // Inserir contexto ao interpretador
    _interpreter->SetExternalContext(kTfLiteEdgeTpuContext, _edgeTpu.get());
    _interpreter->SetNumThreads(1);

    if(_interpreter->AllocateTensors() != kTfLiteOk)
    {
        LOG(ERROR)<<"Failed to allocate tensors."<<std::endl;
        exit(-1);
    }
    LOG(INFO)<<"Success allocate tensors."<<std::endl;

    // Construção de input e output do interpretador
    const auto& dimensions = _interpreter->tensor(_interpreter->inputs()[0])->dims;

    _inputHeight = dimensions->data[1];
    _inputWidth = dimensions->data[2];
    _inputChannels = dimensions->data[3];

    LOG(INFO)<<"Input shape: (";
    for(size_t i = 0; i<dimensions->size; i++){
        if(i == dimensions->size - 1)
        {
            LOG(INFO)<<dimensions->data[i]<<")"<<std::endl;
        }else{
            LOG(INFO)<<dimensions->data[i]<<",";
        }
    }

    std::chrono::duration<double, std::milli> stop_time = std::chrono::steady_clock::now() - start_time;

    std::ostringstream time_caption;

    time_caption << std::fixed << std::setprecision(2) << stop_time.count() << " ms, " << 1000.0 / stop_time.count() << "FPS";

    LOG(INFO)<<"Time for loader interpreter: "<<time_caption.str()<<std::endl;

}
std::unique_ptr<std::vector<Bbox>> Detector::RunInference(const std::vector<uint8_t> &inputData, std::chrono::duration<double, std::milli> &timeSpan)
{
    const auto& start_time = std::chrono::steady_clock::now();

    uint8_t* input = _interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, inputData.data(), inputData.size());

    _interpreter->Invoke();

    float* locations = _interpreter->typed_output_tensor<float>(0);
    float *classes = _interpreter->typed_output_tensor<float>(1);
    float *confidences = _interpreter->typed_output_tensor<float>(2);
    int numDetections = (int)*_interpreter->typed_output_tensor<float>(3);

    LOG(INFO)<<"Num of detections: "<<numDetections<<std::endl;

    auto result = std::make_unique<std::vector<Bbox>>();

    for(auto i = 0; i<numDetections; i++)
    {
        if(confidences[i] >= 0.5)
        {
            auto bbox = std::make_unique<Bbox>();
            float y0 = locations[4 * i + 0];
            float x0 = locations[4 * i + 1];
            float y1 = locations[4 * i + 2];
            float x1 = locations[4 * i + 3];

            bbox->class_id = (int)classes[i];
            bbox->confidence = confidences[i];
            bbox->x = x0;
            bbox->y = y0;
            bbox->width = x1 - x0;
            bbox->height = y1 - y0;
            bbox->center_x = bbox->x + (bbox->width / 2.0);
            bbox->center_y = bbox->y + (bbox->height / 2.0);
            std::cout << "class_id: " << bbox->class_id << std::endl;
            std::cout << "scores  : " << bbox->confidence << std::endl;
            std::cout << "x       : " << bbox->x << std::endl;
            std::cout << "y       : " << bbox->y << std::endl;
            std::cout << "width   : " << bbox->width << std::endl;
            std::cout << "height  : " << bbox->height << std::endl;
            std::cout << "center  : " << bbox->center_x << ", " << bbox->center_y << std::endl;

            result->emplace_back(std::move(*bbox));
        }
    }

    timeSpan = std::chrono::steady_clock::now() - start_time;

    return result;
}
const int Detector::Width() const
{
    return _inputWidth;
}
const int Detector::Height() const
{
    return _inputHeight;
}
const int Detector::Channels() const
{
    return _inputChannels;
}
