#ifndef DETECTOR_H
#define DETECTOR_H

#include "bbox.h"
#include <edgetpu.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#define LOG(x) std::cerr

#if defined(__GNUC__)
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
    #define CDECL __attribute__((cdecl))
#endif

class Detector
{
public:
    Detector();
    const char* GetVersion();

    // Tensor inputs
    const int Width() const;
    const int Height() const;
    const int Channels() const;

    // Construção interpretador
    void BuildInterpreter(const char* _modelPath);

    // Rodar inferência
    std::unique_ptr<std::vector<Bbox>> RunInference(const std::vector<uint8_t>& inputData, std::chrono::duration<double, std::milli> &timeSpan);
private:

    // Declaraçao operadores
    std::unique_ptr<tflite::FlatBufferModel> _model;
    tflite::ops::builtin::BuiltinOpResolver* _resolver;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    std::shared_ptr<edgetpu::EdgeTpuContext> _edgeTpu;

    //Declação Tensores
    TfLiteTensor* _outputLocations = nullptr;
    TfLiteTensor* _outputClasses = nullptr;
    TfLiteTensor* _outputScores = nullptr;
    TfLiteTensor* _numDetections = nullptr;

    // Threshold
    float score_threshold_ = 0.1;

    std::vector<int> input_tensor_shape;

    // Funções internas para construação do modelo e grafo de inferência
    void BuildEdgeTpuInterpreter(const char* modelPath);

    // Input do Modelo
    float _inputHeight;
    float _inputWidth;
    float _inputChannels;
};

extern "C" {
    // chamadas nativas
}

#endif // DETECTOR_H
