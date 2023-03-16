#include <Windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#define _TEMPLATE_FOLDER images
#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define TEMPLATE_FOLDER STRINGIFY(_TEMPLATE_FOLDER)

BITMAPINFO CreateBitmapInfo(int width, int height)
{
	BITMAPINFOHEADER bi;
	bi.biSize = sizeof(BITMAPINFOHEADER);
	bi.biWidth = width;
	bi.biHeight = -height;
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	BITMAPINFO bmi;
	bmi.bmiHeader = bi;
	return bmi;
}

cv::Mat captureScreen()
{
	HWND hDesktopWnd = GetDesktopWindow();
	HDC hDesktopDC = GetDC(hDesktopWnd);
	HDC hCaptureDC = CreateCompatibleDC(hDesktopDC);

	int screenWidth = GetSystemMetrics(SM_CXSCREEN);
	int screenHeight = GetSystemMetrics(SM_CYSCREEN);
	// int screenx = GetSystemMetrics(SM_XVIRTUALSCREEN);
	// int screeny = GetSystemMetrics(SM_YVIRTUALSCREEN);

	int targetWidth = 400;
	int targetHeight = 300;

	HBITMAP hCaptureBitmap = CreateCompatibleBitmap(hDesktopDC, screenWidth, screenHeight);
	BITMAPINFO bmi = CreateBitmapInfo(screenWidth, screenHeight);
	SelectObject(hCaptureDC, hCaptureBitmap);

	
	BitBlt(hCaptureDC, 0, 0, screenWidth, screenHeight, hDesktopDC, 0, 0, SRCCOPY);
	// StretchBlt(hCaptureDC, 0, 0, targetWidth, targetHeight, hDesktopDC, screenx, screeny, screenWidth, screenHeight, SRCCOPY | CAPTUREBLT);

	
	cv::Mat screenCapture(screenHeight, screenWidth, CV_8UC4);
	GetDIBits(hCaptureDC, hCaptureBitmap, 0, screenHeight, screenCapture.data, &bmi, DIB_RGB_COLORS);

	DeleteObject(hCaptureBitmap);
	ReleaseDC(hDesktopWnd, hDesktopDC);
	DeleteDC(hCaptureDC);

	cv::Mat resizedCapture(targetHeight, targetWidth, CV_8UC4);
	cv::resize(screenCapture, resizedCapture, cv::Size(), static_cast<double>(targetWidth) / screenWidth, static_cast<double>(targetHeight) / screenHeight);

	return resizedCapture;
}

std::vector<std::pair<cv::Point, cv::Point>> MultiScaleTemplateMatching(const cv::Mat& targetGray,
                                                                        const cv::Mat& templateGray,
                                                                        const double threshold = 0.75,
                                                                        const double minScale = 0.5,
                                                                        const double maxScale = 2.0,
                                                                        const double scaleStep = 0.5)
{
	std::vector<std::pair<cv::Point, cv::Point>> detections;

	for (double scale = minScale; scale < maxScale; scale += scaleStep)
	{
		cv::Mat resizedTemplate;
		cv::resize(templateGray, resizedTemplate, cv::Size(), scale, scale);

		cv::Mat result;
		cv::matchTemplate(targetGray, resizedTemplate, result, cv::TM_CCOEFF_NORMED);

		cv::Mat mask;
		cv::threshold(result, mask, threshold, 1.0, cv::THRESH_BINARY);

		for (int row = 0; row < mask.rows; ++row)
		{
			for (int col = 0; col < mask.cols; ++col)
			{
				if (mask.at<float>(row, col) > 0)
				{
					detections.emplace_back(cv::Point(col, row),
					                        cv::Point(col + resizedTemplate.cols, row + resizedTemplate.rows));
				}
			}
		}
	}
	return detections;
}

int main()
{
	cv::namedWindow("Screen Capture", cv::WINDOW_NORMAL);
	cv::resizeWindow("Screen Capture", 640, 480);

#pragma region Images Gathering Routine
	extern char* _pgmptr;
	std::filesystem::path executablePath = std::filesystem::path(_pgmptr).parent_path();

	std::filesystem::path templateFolderPath = executablePath / TEMPLATE_FOLDER;

	std::vector<cv::Mat> templateImages;
	int maxTemplateImagesCount = 21;
	int templateImagesCount = 0;

	for (const auto& entry : std::filesystem::directory_iterator(templateFolderPath))
	{
		if (templateImagesCount >= maxTemplateImagesCount)
			break;
		if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".webp" ||
			entry.path().extension() == ".png"))
		{
			cv::Mat templateImage = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

			if (templateImage.empty())
			{
				std::cerr << "Error: Could not read template image " << entry.path() << std::endl;
				continue;
			}

			cv::Mat templateGray;
			cv::cvtColor(templateImage, templateGray, cv::COLOR_BGR2GRAY);
			templateImages.push_back(templateGray);
		}
		templateImagesCount += 1;
	}
#pragma endregion


	while (true)
	{
		cv::Mat screenCapture = captureScreen();
		cv::Mat targetGray;
		cv::cvtColor(screenCapture, targetGray, cv::COLOR_BGR2GRAY);

		bool processTemplates = true;
#pragma region Image Processing Routine
		if (processTemplates)
		{
			for (const auto& templateGray : templateImages)
			{
				std::vector<std::pair<cv::Point, cv::Point>> detections = MultiScaleTemplateMatching(
					targetGray, templateGray);

				for (const auto& detection : detections)
				{
					cv::rectangle(screenCapture, detection.first, detection.second, cv::Scalar(0, 0, 255), 2);
				}
			}
		}
#pragma endregion

		cv::imshow("Screen Capture", screenCapture);

		int key = cv::waitKey(2);
		if (key == 27)
			break;
	}

	cv::destroyAllWindows();

	return 0;
}
