#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
	std::string pb_file_path = "D:/project/openCV_study/opencv480_quick_tutorial/face_detect_demo/opencv_face_detector_uint8.pb";
	std::string pbtxt_file_path = "D:/project/openCV_study/opencv480_quick_tutorial/face_detect_demo/opencv_face_detector.pbtxt";
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(pb_file_path, pbtxt_file_path);
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while (true)
	{
		cap.read(frame);
		if (frame.empty()) {
			break;
		}
		cv::Mat blob = cv::dnn::blobFromImage(frame,1.0,cv::Size(300,300),cv::Scalar(104,177,123),false,false);
		net.setInput(blob); //设置模型输入
		cv::Mat probs = net.forward(); //模型推理
		//1x1xNx7
		cv::Mat detectMat(probs.size[2], probs.size[3],CV_32F,probs.ptr<float>());
		for (int row= 0; row < detectMat.rows; row++)
		{
			float conf = detectMat.at<float>(row, 2);
			if (conf > 0.5)
			{
				float x1 = detectMat.at<float>(row, 3) * frame.cols;
				float y1 = detectMat.at<float>(row, 4) * frame.rows;
				float x2 = detectMat.at<float>(row, 5) * frame.cols;
				float y2 = detectMat.at<float>(row, 6) * frame.rows;
				cv::Rect box(x1, y1, x2 - x1, y2 - y1);
				cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
			}
		}
		cv::imshow("OpenCV4.8 DNN 人脸检测演示", frame);
		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}