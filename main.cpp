#include <opencv2/opencv.hpp>
#include <torch/script.h>

struct BoundingBox {
    int classID;
    float score;
    float xmin, ymin;
    float xmax, ymax;

    BoundingBox(){}

    BoundingBox(int classID, float score, float xmin, float ymin, float xmax, float ymax) {
        this->classID = classID;
        this->score = score;
        this->xmin = xmin;
        this->ymin = ymin;
        this->xmax = xmax;
        this->ymax = ymax;
    }

};

const std::vector<std::string> COCO_NAMES = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
                    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
                    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", 
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
                    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", 
                    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", 
                    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
                    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

class DINO {
    private:
        torch::jit::script::Module model;
        static const int height = 800;
        static const int width = 1200;
        static const bool useGPU = true;
    public:
        DINO(std::string modelPath) {
            std::cout << "Load model from " << modelPath << std::endl;
            this->model = torch::jit::load(modelPath);
            if (DINO::useGPU)
                this->model.to(at::kCUDA);
            std::cout << "Load model successfully" << std::endl;
        }

        std::vector<BoundingBox> forward(cv::Mat &image, float threshold = 0.5) {
            c10::InferenceMode guard(true);
            cv::Mat input;
            cv::resize(image, input, cv::Size(this->width, this->height));
            cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
            float scale_x = (float)image.cols / input.cols;
            float scale_y = (float)image.rows / input.rows;
    
            input.convertTo(input, CV_32FC3);
            torch::Tensor inputTensor = torch::from_blob(input.data, {input.rows, input.cols, 3}, c10::kFloat);
            if (DINO::useGPU)
                inputTensor = inputTensor.to(at::kCUDA);
            inputTensor = inputTensor.permute({2, 0, 1}); 
    
            std::vector<torch::Dict<std::string, torch::Tensor>> inputs(1, torch::Dict<std::string, torch::Tensor>());
            inputs[0].insert("image", inputTensor);

            std::vector<torch::jit::IValue> inputsToNet = {inputs};

            auto outputs = model(inputsToNet);
            
            auto tuple = outputs.toTuple()->elements();
            auto boxes = tuple[0].toTensor();
            auto scores = tuple[1].toTensor();
            auto classes = tuple[2].toTensor(); 

            std::vector<BoundingBox> results;
            for (int i = 0; i < boxes.sizes()[0]; ++i) {
                float score = scores[i].item().toFloat();
                int classID = classes[i].item().toInt();
                float xmin = boxes[i][0].item().toFloat() * scale_x;
                float ymin = boxes[i][1].item().toFloat() * scale_y;
                float xmax = boxes[i][2].item().toFloat() * scale_x;
                float ymax = boxes[i][3].item().toFloat() * scale_y;
                if (score >= threshold)
                    results.push_back(BoundingBox(classID, score, xmin, ymin, xmax, ymax));
            }

            return results;
        }

        
};

int main(int nargs, char **args) {
    if (nargs != 3) {
        std::cout << "Usage: " << args[0] << " <model path> <image input>" << std::endl;
        return -1;
    }

    std::string modelPath(args[1]);
    std::string imagePath(args[2]);
    
    cv::Mat image = cv::imread(imagePath);

    DINO detector(modelPath);

    auto boxes = detector.forward(image);
    
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    float fontScale = 0.6;
    int thickness = 1;
    cv::Scalar color(255, 0, 0);

    for (auto box: boxes) {
        std::string label = COCO_NAMES[box.classID];
        cv::Rect rect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);
        cv::rectangle(image, rect, color, thickness);
        cv::Point textOrg(box.xmin, box.ymin);

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;

        cv::rectangle(image, textOrg + cv::Point(0, baseline), textOrg + cv::Point(textSize.width, -textSize.height), color, -1);
        cv::putText(image, label, textOrg, fontFace, fontScale, cv::Scalar(200, 200, 200), thickness);
    }

    cv::imshow("DINO Detection", image);
    cv::waitKey(0);

    return 0;
}
    
