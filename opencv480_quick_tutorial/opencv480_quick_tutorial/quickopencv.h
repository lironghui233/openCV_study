#pragma once

#include <opencv2/opencv.hpp>

class QuickDemo {
public:
	void read_and_show_Demo();								//ͼ���ȡ����ʾ
	void colorSpace_Demo(cv::Mat& image);					//ͼ��ɫ�ʿռ�ת��
	void mat_creation_Demo(cv::Mat& image);					//ͼ�����Ĵ����븳ֵ
	void pixel_visit_Demo(cv::Mat& image);					//ͼ�����صĶ�д����
	void pixel_operators_Demo(cv::Mat& image);				//ͼ�����ص���������
	void tracking_bar_Demo(cv::Mat& image);					//������
	void key_Demo(cv::Mat& image);							//������Ӧ����
	void color_style_Demo(cv::Mat& image);					//opencv�Դ���ɫ�����
	void bitwise_Demo(cv::Mat& image);						//ͼ�����ص��߼�����
	void channels_Demo(cv::Mat& image);						//ͨ��������ϲ�
	void inrange_Demo(cv::Mat& image);						//ͼ��ɫ�ʿռ�ת��
	void pixel_statistic_Demo(cv::Mat& image);				//ͼ������ֵͳ��	
	void drawing_Demo(cv::Mat& image);						//ͼ�񼸺���״����
	void random_drawing_Demo(cv::Mat& image);				//������������ɫ
	void polyline_drawing_Demo(cv::Mat& image);				//�������������
	void mouse_drawing_Demo(cv::Mat& image);				//����������Ӧ
	void norm_Demo(cv::Mat& image);							//ͼ����������ת�����һ��
	void resize_Demo(cv::Mat& image);						//ͼ���������ֵ
	void flip_Demo(cv::Mat& image);							//ͼ��ת
	void rotate_Demo(cv::Mat& image);						//ͼ����ת	
	void video_Demo(cv::Mat& image);						//��Ƶ����ͷʹ�ã���Ƶ������
	void histogram_Demo(cv::Mat& image);					//ͼ��ֱ��ͼ
	void histogram_2d_Demo(cv::Mat& image);					//��άֱ��ͼ
	void histogram_eq_Demo(cv::Mat& image);					//ֱ��ͼ���⻯
	void blur_Demo(cv::Mat& image);							//ͼ��������
	void gaussian_blur_Demo(cv::Mat& image);				//��˹ģ��
	void bifilter_Demo(cv::Mat& image);						//��˹˫��ģ��
};