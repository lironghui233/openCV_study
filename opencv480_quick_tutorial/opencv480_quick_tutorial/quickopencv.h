#pragma once

#include <opencv2/opencv.hpp>

class QuickDemo {
public:
	void read_and_show_Demo();								//图像读取与显示
	void colorSpace_Demo(cv::Mat& image);					//图像色彩空间转换
	void mat_creation_Demo(cv::Mat& image);					//图像对象的创建与赋值
	void pixel_visit_Demo(cv::Mat& image);					//图像像素的读写操作
	void pixel_operators_Demo(cv::Mat& image);				//图像像素的算术操作
	void tracking_bar_Demo(cv::Mat& image);					//滚动条
	void key_Demo(cv::Mat& image);							//键盘响应操作
	void color_style_Demo(cv::Mat& image);					//opencv自带颜色表操作
	void bitwise_Demo(cv::Mat& image);						//图像像素的逻辑操作
	void channels_Demo(cv::Mat& image);						//通道分离与合并
	void inrange_Demo(cv::Mat& image);						//图像色彩空间转换
	void pixel_statistic_Demo(cv::Mat& image);				//图像像素值统计	
	void drawing_Demo(cv::Mat& image);						//图像几何形状绘制
	void random_drawing_Demo(cv::Mat& image);				//随机数与随机颜色
	void polyline_drawing_Demo(cv::Mat& image);				//多边形填充与绘制
	void mouse_drawing_Demo(cv::Mat& image);				//鼠标操作与响应
	void norm_Demo(cv::Mat& image);							//图像像素类型转换与归一化
	void resize_Demo(cv::Mat& image);						//图像缩放与插值
	void flip_Demo(cv::Mat& image);							//图像翻转
	void rotate_Demo(cv::Mat& image);						//图像旋转	
	void video_Demo(cv::Mat& image);						//视频摄像头使用，视频处理保存
	void histogram_Demo(cv::Mat& image);					//图像直方图
	void histogram_2d_Demo(cv::Mat& image);					//二维直方图
	void histogram_eq_Demo(cv::Mat& image);					//直方图均衡化
	void blur_Demo(cv::Mat& image);							//图像卷积操作
	void gaussian_blur_Demo(cv::Mat& image);				//高斯模糊
	void bifilter_Demo(cv::Mat& image);						//高斯双边模糊
};