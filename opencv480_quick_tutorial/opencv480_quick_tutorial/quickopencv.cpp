#include "quickopencv.h"
#include <algorithm> 

void QuickDemo::read_and_show_Demo() {
	//read image
	cv::Mat src = cv::imread("C:/Users/70756/Desktop/1.png");
	if (src.empty()) {
		printf("could not load image...");
		return;
	}
	//create window with custom features
	cv::namedWindow("输入窗口", cv::WINDOW_FREERATIO);
	//show image
	cv::imshow("输入窗口", src);
}

void QuickDemo::colorSpace_Demo(cv::Mat& image) {
	cv::Mat gray, hsv;
	//transform color
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV); //transfrom to hsv 
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); //transfrom to gray 

	//show image
	cv::imshow("HSV", hsv);
	cv::imshow("GRAY", gray);

	//save image
	cv::imwrite("C:/Users/70756/Desktop/hsv.png", hsv);
	cv::imwrite("C:/Users/70756/Desktop/gray.png", gray);
}

void QuickDemo::mat_creation_Demo(cv::Mat& image)
{
	cv::Mat m1, m2;
	//deep copy
	m1 = image.clone();
	//deep copy
	image.copyTo(m2);

	//create empty image
	cv::Mat m3 = cv::Mat::zeros(cv::Size(512,512), CV_8UC3);
	//create one image
	//cv::Mat m3 = cv::Mat::ones(cv::Size(8, 8), CV_8UC1);
	//mat init
	m3 = cv::Scalar(255,0,0);
	//mat attribute
	std::cout << "width:" << m3.cols << "  height:" << m3.rows << " channel:" << m3.channels() << std::endl;
	//std::cout << m3 << std::endl;

	//show image
	cv::imshow("自定义图像", m3);

	//shallow copy
	cv::Mat m4 = m3;
	m4 = cv::Scalar(0, 255, 0);
	cv::imshow("show1", m3);
	cv::imshow("show2", m4);

	//deep copy
	cv::Mat m5= m4.clone();
	//m4.copyTo(m5);
	m5 = cv::Scalar(0, 0, 255);
	cv::imshow("show3", m4);
	cv::imshow("show4", m5);
}

void QuickDemo::pixel_visit_Demo(cv::Mat& image)
{
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();

	//for (int row = 0; row < h; row++)
	//{
	//	for (int col = 0; col < w; col++)
	//	{
	//		//gray image
	//		if (dims == 1) {
	//			//revert operation
	//			int pv = image.at<uchar>(row,col);
	//			image.at<uchar>(row, col) = 255 - pv;
	//		}
	//		//color image
	//		if (dims == 3) {
	//			//revert operation
	//			cv::Vec3b bgr = image.at<cv::Vec3b>(row, col);
	//			image.at<cv::Vec3b>(row, col)[0] = 255 - bgr[0];
	//			image.at<cv::Vec3b>(row, col)[1] = 255 - bgr[1];
	//			image.at<cv::Vec3b>(row, col)[2] = 255 - bgr[2];
	//		}
	//	}
	//}

	for (int row = 0; row < h; row++)
	{
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			//gray image
			if (dims == 1) {
				//revert operation
				int pv = *current_row;
				*current_row++ = 255 - pv;
			}
			//color image
			if (dims == 3) {
				//revert operation
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}
		}
	}

	cv::imshow("像素读写演示", image);
}

void QuickDemo::pixel_operators_Demo(cv::Mat& image) {
	imshow("原图", image);

	cv::Mat dst;
	dst = image - cv::Scalar(50, 50, 50);
	imshow("减法操作", dst);

	dst = image + cv::Scalar(50, 50, 50);
	imshow("加法操作", dst);

	cv::Mat m = cv::Mat::zeros(image.size(), image.type());
	m = cv::Scalar(2, 2, 2);
	cv::multiply(image, m, dst);
	//dst = image * cv::Scalar(0.3, 0.3, 0.3);
	imshow("乘法操作", dst);

	dst = image / cv::Scalar(50, 50, 50);
	imshow("除法操作", dst);

	//pixel multiplication operator
	dst = cv::Mat::zeros(image.size(), image.type());
	m = cv::Mat::zeros(image.size(), image.type());
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++)
	{
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++)
		{
			cv::Vec3b p1 = image.at<cv::Vec3b>(row, col);
			cv::Vec3b p2 = m.at<cv::Vec3b>(row, col);
			dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(p1[0] + p2[0]);
			dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(p1[1] + p2[1]);
			dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(p1[2] + p2[2]);
		}
	}
	imshow("加法操作2", dst);

	//function add 
	add(image, m, dst);
	imshow("加法操作3", dst);

	//function sub 
	subtract(image, m, dst);
	imshow("减法操作2", dst);

	//function div 
	divide(image, m, dst);
	imshow("除法操作2", dst);
}

#define CLAMP(value, lower, upper) ((value) < (lower) ? (lower) : ((value) > (upper) ? (upper) : (value))) 
static void on_light_change(int pos, void *src)
{
	cv::Mat* image = static_cast<cv::Mat*>(src);
	if (image->empty()) {
		std::cerr << "Error: image is empty!" << std::endl;
		return;
	}

	int lightness = pos - 128; // 将滑动条的值映射到亮度变化，假设中间值 128 不改变亮度  
	lightness = CLAMP(lightness, -255, 255); // 限制亮度变化范围  

	cv::Mat dst = cv::Mat::zeros(image->size(), image->type());
	cv::Mat m = cv::Mat::zeros(image->size(), image->type());
	m.setTo(cv::Scalar(lightness, lightness, lightness)); // 创建亮度矩阵  

	//cv::add(*image, m, dst); // 调整亮度  
	cv::addWeighted(*image, 1.0, m, 0, pos, dst);
	cv::imshow("亮度与对比度调整", dst); // 显示结果  
}

static void on_constract_change(int pos, void* src)
{
	cv::Mat* image = static_cast<cv::Mat*>(src);
	if (image->empty()) {
		std::cerr << "Error: image is empty!" << std::endl;
		return;
	}

	int lightness = pos - 128; // 将滑动条的值映射到亮度变化，假设中间值 128 不改变亮度  
	lightness = CLAMP(lightness, -255, 255); // 限制亮度变化范围  

	cv::Mat dst = cv::Mat::zeros(image->size(), image->type());
	cv::Mat m = cv::Mat::zeros(image->size(), image->type());
	m.setTo(cv::Scalar(lightness, lightness, lightness)); // 创建亮度矩阵  

	double constract = pos / 200.0;
	cv::addWeighted(*image, constract, m, 0.0, pos, dst);	//调整对比度
	cv::imshow("亮度与对比度调整", dst); // 显示结果  
}

void QuickDemo::tracking_bar_Demo(cv::Mat& image) 
{
	cv::namedWindow("亮度与对比度调整", cv::WINDOW_AUTOSIZE);
	int max_light_value = 255; // 通常滑动条的最大值会设置为 255  
	int initial_lightness = 128; // 初始亮度调整值，对应于不改变亮度  
	int max_constract_value = 200;
	int initial_constract_value = 100; //默认对比度
	cv::createTrackbar("Value Bar:", "亮度与对比度调整", &initial_lightness, max_light_value, on_light_change, (void*) & image);
	cv::createTrackbar("Constract Bar:", "亮度与对比度调整", &initial_constract_value, max_constract_value, on_constract_change, (void*)&image);
	
}

void QuickDemo::key_Demo(cv::Mat& image)
{
	cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
	while (true)
	{
		int c = cv::waitKey(100);
		if (c == 27) //Esc
			break;
		if (c == 49) {
			std::cout << "you enter key #1" << std::endl;
			cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
		}
		if (c == 50) {
			std::cout << "you enter key #2" << std::endl;
			cv::cvtColor(image, dst, cv::COLOR_BGR2HSV);
		}
		if (c == 51) {
			std::cout << "you enter key #3" << std::endl;
			dst = cv::Scalar(50,50,50);
			cv::add(image, dst, dst);
		}
		cv::imshow("键盘响应", dst);
	}
}

void QuickDemo::color_style_Demo(cv::Mat& image)
{
	int colormap[] = {
		cv::COLORMAP_AUTUMN,
		cv::COLORMAP_BONE,
		cv::COLORMAP_JET,
		cv::COLORMAP_WINTER,
		cv::COLORMAP_RAINBOW,
		cv::COLORMAP_OCEAN,
		cv::COLORMAP_SUMMER,
		cv::COLORMAP_SPRING,
		cv::COLORMAP_COOL,
		cv::COLORMAP_HSV,
		cv::COLORMAP_PINK,
		cv::COLORMAP_HOT,
		cv::COLORMAP_PARULA,
		cv::COLORMAP_MAGMA,
		cv::COLORMAP_INFERNO,
		cv::COLORMAP_PLASMA,
		cv::COLORMAP_VIRIDIS,
		cv::COLORMAP_CIVIDIS,
		cv::COLORMAP_TWILIGHT,
		cv::COLORMAP_TWILIGHT_SHIFTED,
		cv::COLORMAP_TURBO,
		cv::COLORMAP_DEEPGREEN,
	};

	cv::Mat dst;
	int index = 0;
	while (true) {
		int c = cv::waitKey(2000);
		if (c == 27) { // Esc
			break;
		}
		cv::applyColorMap(image, dst, colormap[index % 22]);
		index++;
		cv::imshow("颜色风格", dst);
	}
}

void QuickDemo::bitwise_Demo(cv::Mat& image) {
	cv::Mat m1 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
	cv::Mat m2 = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
	cv::rectangle(m1, cv::Rect(100, 100, 80, 80), cv::Scalar(255,255,0),-1, cv::LINE_8, 0);
	cv::rectangle(m2, cv::Rect(150, 150, 80, 80), cv::Scalar(0, 255, 255), -1, cv::LINE_8, 0);
	cv::imshow("m1", m1);
	cv::imshow("m2", m2);
	cv::Mat dst;
	cv::bitwise_and(m1, m2, dst);
	cv::imshow("像素位and", dst);
	cv::bitwise_or(m1, m2, dst);
	cv::imshow("像素位or", dst);
	cv::bitwise_not(image, dst);
	cv::imshow("像素位not", dst);
	cv::bitwise_xor(m1, m2, dst);
	cv::imshow("像素位xor", dst);
}

void QuickDemo::channels_Demo(cv::Mat& image) 
{
	std::vector<cv::Mat> mv;
	cv::split(image, mv);
	cv::imshow("蓝色", mv[0]);
	cv::imshow("绿色", mv[1]);
	cv::imshow("红色", mv[2]);

	cv::Mat dst;
	//mv[0] = 0;
	mv[1] = 0;
	mv[2] = 0;
	cv::merge(mv,dst);
	cv::imshow("合并", dst);

	int from_to[] = {0,2,1,1,2,0};
	cv::mixChannels(&image, 1, &dst, 1, from_to, 3);
	cv::imshow("通道混合", dst);
}

void QuickDemo::inrange_Demo(cv::Mat& image)
{
	cv::Mat hsv;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask;
	inRange(hsv, cv::Scalar(35, 43, 46), cv::Scalar(77, 255, 255), mask); //提取绿色区域mask
	cv::imshow("mask", mask);

	cv::Mat redback = cv::Mat::zeros(image.size(), image.type());
	redback = cv::Scalar(40, 40, 200);
	cv::bitwise_not(mask, mask); //mask区域取反
	cv::imshow("mask not", mask);
	image.copyTo(redback, mask);
	cv::imshow("ROI区域提取", redback);
}

void QuickDemo::pixel_statistic_Demo(cv::Mat& image)
{
	double minv, maxv;
	cv::Point minLoc, maxLoc;
	std::vector<cv::Mat> mv;
	cv::split(image, mv);
	for (int i = 0; i < mv.size(); i++) {
		cv::minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, cv::Mat());
		std::cout << "No. channels:" << i << "  min value:" << minv << "  max value:" << maxv << std::endl;
	}

	cv::Mat mean, stddev;
	cv::meanStdDev(image, mean, stddev);	//求方差，方差小图片差异化小（纯色图片？），方差大图片差异化大（图片内容很丰富）
	std::cout << "means:" << mean <<  std::endl;
	std::cout << " stddev:" << stddev << std::endl;
}

void QuickDemo::drawing_Demo(cv::Mat& image) 
{
	cv::Rect rect;
	rect.x = 100;
	rect.y = 100;
	rect.width = 100;
	rect.height = 100;
	cv::Mat bg = cv::Mat::zeros(image.size(), image.type());
	cv::rectangle(bg, rect, cv::Scalar(0,0,255), 2, 8, 0); //绘制长方形
	cv::circle(bg, cv::Point(350, 400), 15, cv::Scalar(255,0,0), 2, 8, 0);	//绘制圆形
	cv::line(bg, cv::Point(100,100),cv::Point(35,400), cv::Scalar(0, 255, 0),2, cv::LINE_AA,0);	//绘制直线
	cv::RotatedRect rrt;
	rrt.center = cv::Point(200, 200);
	rrt.size = cv::Size(100, 200);
	rrt.angle = 0.0;
	cv::ellipse(bg, rrt, cv::Scalar(0, 255, 255), 2, 8);  //绘制椭圆
	cv::Mat dst;
	cv::addWeighted(image, 0.7, bg, 0.3, 0, dst);
	imshow("绘几何图", dst);
}

void QuickDemo::random_drawing_Demo(cv::Mat& image)
{
	cv::Mat canvas = cv::Mat::zeros(cv::Size(512,512),CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	cv::RNG rng(12345);
	while (true)
	{
		int c = cv::waitKey(10);
		if (c == 27) {
			break;
		}
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		// canvas = cv::Scalar(0 ,0, 0);
		cv::line(canvas, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), 4, cv::LINE_AA, 0);
		cv::imshow("随机绘制演示", canvas);
	}
}

void QuickDemo::polyline_drawing_Demo(cv::Mat& image)
{
	cv::Mat canvas = cv::Mat::zeros(cv::Size(512,512), CV_8UC3);
	cv::Point p1(100, 100);
	cv::Point p2(350, 100);
	cv::Point p3(450, 280);
	cv::Point p4(320, 450);
	cv::Point p5(80, 400);
	std::vector<cv::Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
	//cv::polylines(canvas, pts, true, cv::Scalar(0, 0, 255), 2, 8, 0);
	//cv::fillPoly(canvas, pts, cv::Scalar(0, 0, 255), cv::LINE_AA, 0);
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(pts);
	cv::drawContours(canvas, contours, -1, cv::Scalar(255,0,0), -1);
	cv::imshow("多边形绘制", canvas);
}

cv::Point sp(-1, -1);
cv::Point ep(-1, -1);
cv::Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata) {
	cv::Mat image = *((cv::Mat*)userdata);
	if (event == cv::EVENT_LBUTTONDOWN) {
		sp.x = x;
		sp.y = y;
		std::cout << "start point:" << sp << std::endl;
	}
	else if (event == cv::EVENT_LBUTTONUP) {
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
			cv::Rect box(sp.x, sp.y, dx, dy);
			temp.copyTo(image);
			cv::imshow("ROI区域", image(box));
			cv::rectangle(image, box, cv::Scalar(0,0,255), 2, 8, 0);
			cv::imshow("鼠标绘制", image);
			//ready for next drawing
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == cv::EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0)
		{
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				cv::Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2, 8, 0);
				cv::imshow("鼠标绘制", image);
			}
		}
		
	}
}

void QuickDemo::mouse_drawing_Demo(cv::Mat& image)
{
	cv::namedWindow("鼠标绘制", cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback("鼠标绘制", on_draw, (void*)&image);
	cv::imshow("鼠标绘制", image);
	temp = image.clone();
}

void QuickDemo::norm_Demo(cv::Mat& image)
{
	cv::Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);
	std::cout << image.type() << std::endl;
	cv::normalize(image, dst, 1.0, 0, cv::NORM_MINMAX);
	std::cout << dst.type() << std::endl;
	cv::imshow("图像数据归一化", dst);
}

void QuickDemo::resize_Demo(cv::Mat& image)
{
	cv::imshow("row", image);
	cv::Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	cv::resize(image, zoomin, cv::Size(w/2, h/2),0,0,cv::INTER_LINEAR);
	cv::imshow("zoomin", zoomin);
	cv::resize(image, zoomout, cv::Size(w * 1.2, h * 1.2), 0, 0, cv::INTER_LINEAR);
	cv::imshow("zoomout", zoomout);
}

void QuickDemo::flip_Demo(cv::Mat& image)
{
	cv::Mat dst;
	//cv::flip(image, dst, 0); //上下翻转
	//cv::flip(image, dst, 1); //左右翻转
	cv::flip(image, dst, -1); //对角线翻转
	cv::imshow("图像翻转", dst);
}

void QuickDemo::rotate_Demo(cv::Mat& image)
{
	cv::Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = cv::getRotationMatrix2D(cv::Point(w/2, h/2), 45, 1.0);
	std::cout << "M: " << M << std::endl;
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = w * cos + h * sin;
	int nh = w * sin + h * cos;
	M.at<double>(0, 2) = M.at<double>(0, 2) + (nw / 2 - w / 2);
	M.at<double>(1, 2) = M.at<double>(1, 2) + (nh / 2 - h / 2);
	cv::warpAffine(image, dst, M, cv::Size(nw,nh), cv::INTER_LINEAR, 0, cv::Scalar(255, 255, 0));
	cv::imshow("旋转演示", dst);
}

void QuickDemo::video_Demo(cv::Mat& image)
{
	cv::VideoCapture capture("C:/Users/70756/Desktop/2.mp4");

	//property
	int frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(cv::CAP_PROP_FRAME_COUNT);
	double fps = capture.get(cv::CAP_PROP_FPS);
	std::cout << "frame width:" << frame_width << std::endl;
	std::cout << "frame height:" << frame_height << std::endl;
	std::cout << "FPS:" << fps << std::endl;
	std::cout << "Number pf Frames:" << count << std::endl;

	//save
	cv::VideoWriter writer("C:/Users/70756/Desktop/3.mp4", capture.get(cv::CAP_PROP_FOCUS), fps, cv::Size(frame_width, frame_height), true);

	cv::Mat frame;
	while (true) 
	{
		capture.read(frame);
		cv::flip(frame, frame, 1);
		if (frame.empty())
		{
			break;
		}
		cv::imshow("frame", frame);
		// TODO:do something
		colorSpace_Demo(frame);

		//save
		writer.write(frame);

		int c = cv::waitKey(10);
		if (c == 27) {	//Eec
			break;
		}
		/*if (c == 49) {
			std::cout << "you enter key #1" << std::endl;
			cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
		}
		if (c == 50) {
			std::cout << "you enter key #2" << std::endl;
			cv::cvtColor(image, dst, cv::COLOR_BGR2HSV);
		}
		if (c == 51) {
			std::cout << "you enter key #3" << std::endl;
			dst = cv::Scalar(50, 50, 50);
			cv::add(image, dst, dst);
		}*/
	}
	//release
	capture.release();
	writer.release();
}

void QuickDemo::histogram_Demo(cv::Mat& image)
{
	// 三通道分离
	std::vector<cv::Mat> bgr_plane;
	cv::split(image, bgr_plane);
	// 定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };
	cv::Mat b_hist;
	cv::Mat g_hist;
	cv::Mat r_hist;
	// 计算Blue，Green，Red通道的直方图
	cv::calcHist(&bgr_plane[0], 1, 0, cv::Mat(), b_hist, 1, bins, ranges);
	cv::calcHist(&bgr_plane[1], 1, 0, cv::Mat(), g_hist, 1, bins, ranges);
	cv::calcHist(&bgr_plane[2], 1, 0, cv::Mat(), r_hist, 1, bins, ranges);
	// 显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w/bins[0]);
	cv::Mat histImage = cv::Mat::zeros(hist_h, hist_w, CV_8UC3);
	// 归一化直方图数据
	cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	// 绘制直方图曲线
	for (int i = 0; i < bins[0]; i++)
	{
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))), cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))), cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))), cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	// 显示直方图
	cv::namedWindow("Histogram Demo", cv::WINDOW_AUTOSIZE);
	cv::imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_2d_Demo(cv::Mat& image)
{
	// 2D 直方图
	cv::Mat hsv, hs_hist;
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
	int hbins = 20, sbins = 32;
	int hist_bins[] = { hbins, sbins };
	float h_range[] = { 0,180 };
	float s_range[] = { 0,256 };
	const float* hs_ranges[] = { h_range,s_range };
	int hs_channels[] = {0,1};
	cv::calcHist(&hsv, 1, hs_channels, cv::Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
	double maxVal = 0;
	cv::minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
	int scale = 10;
	cv::Mat hist2d_image = cv::Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);
	for (int h = 0; h < hbins; h++)
	{
		for(int s = 0; s < sbins; s++)
		{
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			cv::rectangle(hist2d_image, cv::Point(h*scale, s*scale), cv::Point((h+1)*scale -1,(s+1)*scale-1),cv::Scalar::all(intensity),-1);
		}
	}
	cv::applyColorMap(hist2d_image, hist2d_image, cv::COLORMAP_JET);
	cv::imshow("H-S Histogram", hist2d_image);
	cv::imwrite("C:/Users/70756/Desktop/hist_2d.png", hist2d_image);
}

void QuickDemo::histogram_eq_Demo(cv::Mat& image)
{
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::imshow("灰度图像", gray);
	cv::Mat dst;
	cv::equalizeHist(gray, dst);
	cv::imshow("直方图均衡化演示", dst);
}

void QuickDemo::blur_Demo(cv::Mat& image)
{
	cv::Mat dst;
	cv::blur(image, dst, cv::Size(33, 33), cv::Point(-1,-1)); //均值卷积操作
	cv::imshow("卷积操作", dst);
}

void QuickDemo::gaussian_blur_Demo(cv::Mat& image)
{
	cv::Mat dst;
	cv::GaussianBlur(image, dst, cv::Size(0,0), 15);
	cv::imshow("高斯模糊", dst);
}

void QuickDemo::bifilter_Demo(cv::Mat& image)
{
	cv::Mat dst;
	cv::bilateralFilter(image, dst, 0, 100, 10);
	cv::imshow("高斯双边模糊", dst);
}