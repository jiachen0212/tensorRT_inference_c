// trt_inference.cpp
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <NvInfer.h>
#include <NvOnnxParser.h>
​
#include <cuda_runtime.h>
// 导入utils
#include "utils.h"
// onnx.model参数保持一致
int INPUT_H=224;
int INPUT_W=224;
int OUTPUT_SIZE=1000;
// batch_size
int batchSize = 4;
​
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) override {
        using namespace std;
        string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        cerr << s << ": " << msg << endl;
    }
};
​
/* 自销毁定义，使用方便 */
template<typename T>
struct Destroy {
    void operator()(T *t) const {
        t->destroy();
    }
};
​
// Optional : Print dimensions as string
std::string printDim(const nvinfer1::Dims & d) {
    using namespace std;
    ostringstream oss;
    for (int j = 0; j < d.nbDims; ++j) {
        oss << d.d[j];
        if (j < d.nbDims - 1)
            oss << "x";
    }
    return oss.str();
}
​
// 从onxx中读取模型 
nvinfer1::ICudaEngine *createCudaEngine(const std::string &onnxFileName, nvinfer1::ILogger &logger, int batchSize) {
    using namespace std;
    using namespace nvinfer1;
​
    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(logger)};
    // 定义网络类型 
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)};
    // onnxParse解析器, 将onnx模型转化为engine 
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
            nvonnxparser::createParser(*network, logger)};
​
    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");
​
    // Create Optimization profile and set the batch size
    // This profile will be valid for all images whose size falls in the range
    // but TensorRT will optimize for {batchSize, 3, 422,422}
    // We do not need to check the return of setDimension and addOptimizationProfile here as all dims are explicitly set
    IOptimizationProfile *profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4{batchSize, 3, 1, 1});
    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4{batchSize, 3, 422,422});
    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4{batchSize, 3, INPUT_H, INPUT_W});
    
    // Create a builder configuration object.
    unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    return builder->buildEngineWithConfig(*network, *config); 
}
// 前向输出并放入cuda
void launchInference(nvinfer1::IExecutionContext *context, cudaStream_t stream, float * input,
                     float * output, void **buffers, int batchSize) {
    
    int inputId = 0, outputId = 1;
​
    cudaMemcpyAsync(buffers[inputId], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
​
    context->enqueueV2(buffers, stream, nullptr);
​
    cudaMemcpyAsync(output, buffers[outputId], batchSize * OUTPUT_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
}
​
​
int main(int argc, char **argv)
{
    using namespace std;
    using namespace nvinfer1;
​
    Logger logger;
    logger.log(ILogger::Severity::kINFO, "C++ TensorRT (almost) minimal example !!! ");
    logger.log(ILogger::Severity::kINFO, "Creating engine ...");
    // onnx解析成engine
    // In order to create an object of type IExecutionContext, first create an object of type ICudaEngine (the engine).
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(createCudaEngine("model.onnx", logger, batchSize));  
​
    // 从二进制文件trt中取模型流并生成engine 
    ifstream file("fp_mix.trt", ios::binary);
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    unique_ptr<IRuntime, Destroy<IRuntime>> runtime(createInferRuntime(logger));
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine(runtime->deserializeCudaEngine(trtModelStream, size, nullptr));
​
   // 查看engine输入输出维度 
   // Optional : Print all bindings : name + dims + dtype
    cout << "=============\nBindings :\n";
    int n = engine->getNbBindings();
    for (int i = 0; i < n; ++i) {
        Dims d = engine->getBindingDimensions(i);
        cout << i << " : " << engine->getBindingName(i) << " : dims=" << printDim(d);
        cout << " , dtype=" << (int) engine->getBindingDataType(i) << " ";
        cout << (engine->bindingIsInput(i) ? "IN" : "OUT") << endl;
    }
    cout << "=============\n\n";
​
    /* In order to run inference, use the interface IExecutionContext */
    logger.log(ILogger::Severity::kINFO, "Creating context ...");
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context(engine->createExecutionContext());
    context->setBindingDimensions(0, Dims4(batchSize, 3, INPUT_H, INPUT_W));
​
    //  Create an asynchronous stream 
    cudaStream_t stream;
    cudaStreamCreate(&stream);
​
    // input output buffer  
    float data[batchSize * 3 * INPUT_W * INPUT_H];
    float prob[batchSize * OUTPUT_SIZE];
    // cuda buffer 
    void* buffers[2]{0};
    cudaMalloc(&buffers[0], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * OUTPUT_SIZE * sizeof(float));
​
    logger.log(ILogger::Severity::kINFO, "---------------------prepare ok-------------------------");
​
    // 输入image_path 
    std::string img_dir = std::string(argv[1]);
    std::vector<std::string> file_names;
    // utils.h 
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_imgs_in_dir failed." << std::endl;
        return -1;
    }
    logger.log(ILogger::Severity::kINFO, "---------------------read imgs ok-------------------------");
​
    int fcount = 0;
    for (int f = 0; f < (int)file_names.size(); f++) {
        fcount++;
        if (fcount < batchSize && f + 1 != (int)file_names.size()) continue;
        for (int b = 0; b < fcount; b++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
            if (img.empty()) continue;
            // utils.h
            cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
            int i = 0;
            for (int row = 0; row < INPUT_H; ++row) {
                uchar* uc_pixel = pr_img.data + row * pr_img.step;
                // img_normailze 
                for (int col = 0; col < INPUT_W; ++col) {
                    data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                    data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                    uc_pixel += 3;
                    ++i;
                }
            }
        }
    }
    logger.log(ILogger::Severity::kINFO, "---------------------read imgs ok-------------------------");
    // ​inference 
    cout << "Running the inference !" << endl;
    launchInference(context.get(), stream, data, prob, buffers, batchSize);
    cudaStreamSynchronize(stream);

    // ​output: prob 
    cout << "y = [";
    for (int i = 0; i < batchSize; ++i) {
        for (int j=0;j<OUTPUT_SIZE;j++)
            cout << prob[j]  << ",";
        cout << ";\n";
    }
    cout << " ]" << endl;

    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    return 0;
}