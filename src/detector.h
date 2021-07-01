#ifndef DETECTOR_H
#define DETECTOR_H
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>


class Detector
{
public:
    Detector();
    const char* GetVersion();

    // Construção interpretador
    bool BuildInterpreter(const char* _modelPath, const unsigned int num_of_threads = 1);

    // Rodar inferência

private:

    // Declaraçao operadores
    std::unique_ptr<tflite::FlatBufferModel> _model;
    tflite::ops::builtin::BuiltinOpResolver* _resolver;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    std::shared_ptr<edgetpu::EdgeTpuContext> _edgeTpu;

    //Declação Tensores
    TfLiteTensor* _outputLocations = nullptr;
    TfLiteTensor* _outputClasses = nullptr;
    TfLiteTensor* _output_scores = nullptr;
    TfLiteTensor* _numDetections = nullptr;

    // Threshold
    float score_threshold_ = 0.5;

    std::vector<int> input_tensor_shape;
    size_t input_array_size = 1;

    // Funções internas para construação do modelo e grafo de inferência
    bool BuildInterpreterInternal(const unsigned int num_of_threads);
    bool BuildEdgeTpuInterpreterInternal(const char* modelPath, unsigned int num_of_threads = 1);
    float* GetTensorData(TfLiteTensor& tensor, const int index = 0);
    TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src);
};

#endif // DETECTOR_H
