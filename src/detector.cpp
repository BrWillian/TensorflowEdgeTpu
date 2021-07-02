#include "detector.h"

Detector::Detector()
{

}
const char* Detector::GetVersion()
{
    return "1.0.0";
}
bool Detector::BuildInterpreter(const char *_modelPath)
{
    _model = tflite::FlatBufferModel::BuildFromFile(_modelPath);

    if(_model == nullptr)
    {
        std::cerr<<"Fail to build FlatBufferModel from file: "<<_modelPath<<std::endl;
        return false;
    }

    return BuildEdgeTpuInterpreter(_modelPath);
}
bool Detector::BuildEdgeTpuInterpreter(const char *_modelPath)
{
    const auto& start_time = std::chrono::steady_clock::now();

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
    _interpreter->SetNumThreads(20);

    if(_interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr<<"Failed to allocate tensors."<<std::endl;
        return false;
    }
    std::cout<<"Success allocate tensors."<<std::endl;

    // Construção de input e output do interpretador
    const auto& dimensions = _interpreter->tensor(_interpreter->inputs()[0])->dims;

    _inputHeight = dimensions->data[1];
    _inputWidth = dimensions->data[2];
    _inputChannels = dimensions->data[3];

    std::cout<<"Input shape: (";
    for(size_t i = 0; i<dimensions->size; i++){
        if(i == dimensions->size - 1)
        {
            std::cout<<dimensions->data[i]<<")"<<std::endl;
        }else{
            std::cout<<dimensions->data[i]<<",";
        }
    }

    // Saida do tensor
    _outputLocations = _interpreter->tensor(_interpreter->outputs()[0]);
    _outputClasses = _interpreter->tensor(_interpreter->outputs()[1]);
    _outputScores = _interpreter->tensor(_interpreter->outputs()[2]);
    _numDetections = _interpreter->tensor(_interpreter->outputs()[3]);


    std::chrono::duration<double, std::milli> stop_time = std::chrono::steady_clock::now() - start_time;

    std::ostringstream time_caption;

    time_caption << std::fixed << std::setprecision(2) << stop_time.count() << " ms, " << 1000.0 / stop_time.count() << "FPS";

    std::cout<<"Time for loader interpreter: "<<time_caption.str()<<std::endl;

    return true;
}
std::unique_ptr<std::vector<Bbox>> Detector::RunInference(const std::vector<uint8_t> &inputData, std::chrono::duration<double, std::milli> &timeSpan)
{
    const auto& start_time = std::chrono::steady_clock::now();

    uint8_t* input = _interpreter->typed_input_tensor<uint8_t>(0);
    std::memcpy(input, inputData.data(), inputData.size());

    _interpreter->Invoke();

    const float* locations = GetTensorData(*_outputLocations);
    const float* classes = GetTensorData(*_outputClasses);
    const float* confidences = GetTensorData(*_outputScores);
    const int numDetections = (int)*GetTensorData(*_numDetections);

    auto result = std::make_unique<std::vector<Bbox>>();

    for(size_t i = 0; i<numDetections; i++)
    {
        if(confidences[i] >= score_threshold_)
        {
            std::unique_ptr<Bbox> bbox = std::make_unique<Bbox>();
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
            std::cout << "y       : " << bbox->y << std::endl;

            result->emplace_back(std::move(*bbox));
        }
    }
    timeSpan = std::chrono::steady_clock::now() - start_time;

    return result;
}
float* Detector::GetTensorData(TfLiteTensor &tensor, const int index)
{
    float* result = nullptr;
    auto nelems = 1;
    for(size_t i = 1; i < tensor.dims->size; i++)
    {
        nelems *= tensor.dims->data[i];
    }

    switch (tensor.type)
    {
    case kTfLiteFloat32:
        result = tensor.data.f + nelems * index;
        break;
        std::cerr << "Unmatch tensor type." << std::endl;
    default:
        break;
    }
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
