#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Coarse eye detection using Haar cascade
std::vector<cv::Rect> coarseFind(const cv::Mat& frame) {
    cv::CascadeClassifier eye_cascade;
    eye_cascade.load("haarcascade_eye.xml");  // adjust path as needed
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(
        gray,
        eyes,
        1.05,            // scaleFactor
        2,               // minNeighbors
        0,
        cv::Size(150,150)
    );
    return eyes;
}

// Finds the longest contiguous non-zero segment after the first large incline
std::pair<int,int> longestNonZeroSegment(const std::vector<long long>& arr_in) {
    const long long inclineThresh = 2000;
    int n = (int)arr_in.size();
    if (n < 3) {
        return { -1, -1 };
    }

    std::vector<long long> arr = arr_in;

    // 1) Locate first large incline
    int incline = 0;
    for (int i = 1; i < n-1; ++i) {
        if (arr[i+1] - arr[i] > inclineThresh) {
            incline = i;
            break;
        }
    }

    // 2) Search for longest non-zero segment in arr[incline:]
    int bestStart = 0, bestEnd = n-1;
    int maxLen = 0;
    int tempStart = -1;
    for (int i = incline; i < n; ++i) {
        if (arr[i] != 0) {
            if (tempStart < 0) tempStart = i;
        } else {
            if (tempStart >= 0) {
                int len = i - tempStart;
                if (len > maxLen) {
                    maxLen = len;
                    bestStart = tempStart;
                    bestEnd   = i - 1;
                }
                tempStart = -1;
            }
        }
    }
    // handle tail
    if (tempStart >= 0) {
        int len = n - tempStart;
        if (len > maxLen) {
            bestStart = tempStart;
            bestEnd   = n - 1;
        }
    }

    return { bestStart, bestEnd };
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "No video path provided\n";
        return 1;
    }
    std::string videoName = argv[1];
    std::string videoPath = "videos/" + videoName + ".mp4";
    if (!fs::exists(videoPath)) {
        std::cerr << "Video " << videoPath << " does not exist\n";
        return 1;
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video\n";
        return 1;
    }

    if (!fs::exists("output")) {
        fs::create_directory("output");
    }
    std::ofstream csv("output/eye_data.csv");
    csv << "frame_idx,x,y,w,h,ellipse_x,ellipse_y,ellipse_w,ellipse_h,ellipse_angle\n";

    int frameIdx = 0;
    std::vector<cv::Rect> prevEyes;
    bool hasPrevEyes = false;
    cv::RotatedRect prevEllipse;
    bool hasPrevEllipse = false;

    auto tStart = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        // crop top half
        frame = frame(cv::Range(0, frame.rows / 2), cv::Range::all());

        // detect eyes
        auto eyes = coarseFind(frame);
        if (!eyes.empty()) {
            prevEyes = eyes;
            hasPrevEyes = true;
        } else if (hasPrevEyes) {
            eyes = prevEyes;
        } else {
            ++frameIdx;
            continue;
        }

        int x = eyes[0].x, y = eyes[0].y, w = eyes[0].width, h = eyes[0].height;
        int size = std::max(w, h);
        cv::Mat eyeCrop = frame(cv::Rect(x, y, size, size)).clone();

        cv::Mat eyeGray;
        cv::cvtColor(eyeCrop, eyeGray, cv::COLOR_BGR2GRAY);
        cv::Mat toSave = eyeGray.clone();

        std::vector<int> thresholds = {40, 50, 60};
        std::vector<cv::Mat> maskImages;
        std::vector<cv::RotatedRect> ellipses;
        std::vector<bool> validEllipse;

        for (int thr : thresholds) {
            // threshold + flood-fill
            cv::Mat threshImg;
            cv::threshold(eyeGray, threshImg, thr, 100, cv::THRESH_BINARY_INV);
            cv::Mat flood = threshImg.clone();
            cv::Mat mask = cv::Mat::zeros(threshImg.rows+2, threshImg.cols+2, CV_8U);
            cv::floodFill(flood, mask, cv::Point(0,0), cv::Scalar(255));
            cv::Mat floodInv;
            cv::bitwise_not(flood, floodInv);
            cv::Mat filled;
            cv::bitwise_or(threshImg, floodInv, filled);

            // x-histogram
            cv::Mat sumCols;
            cv::reduce(filled, sumCols, 0, cv::REDUCE_SUM, CV_32S);
            std::vector<long long> xHist(sumCols.cols);
            for (int i = 0; i < sumCols.cols; ++i)
                xHist[i] = sumCols.at<int>(0,i);

            // find segment
            int maxIdx = std::distance(xHist.begin(),
                                      std::max_element(xHist.begin(), xHist.end()));
            long long middle = xHist[maxIdx] / 3;
            std::vector<long long> above(xHist.size(),0);
            for (size_t i = 0; i < xHist.size(); ++i)
                above[i] = (xHist[i] > middle ? xHist[i] : 0LL);

            auto [start, end] = longestNonZeroSegment(above);
            // expand
            while (start > 0 && xHist[start-1] < xHist[start]) start = std::max(0, start-2);
            while (end < (int)xHist.size()-1 && xHist[end+1] < xHist[end])
                end = std::min((int)xHist.size()-1, end+2);
            int center = (start + end) / 2;
            int delta  = end - center;
            if (delta != 0) {
                start = std::max(0, start - int((center-start)*50.0/delta));
                end   = std::min((int)xHist.size()-1,
                                end   + int((end-center)*50.0/delta));
            }

            // mask out-of-range cols
            filled.colRange(0, start).setTo(0);
            filled.colRange(end, filled.cols).setTo(0);
            cv::Mat kClose = cv::Mat::zeros(3,3,CV_8U);
            cv::Mat kOpen  = cv::Mat::ones (3,3,CV_8U);
            cv::morphologyEx(filled, filled, cv::MORPH_CLOSE, kClose);
            cv::morphologyEx(filled, filled, cv::MORPH_OPEN,  kOpen);

            maskImages.push_back(filled);

            // find & clean contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(filled, contours,
                             cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::vector<cv::Point>> cleaned;
            int margin = 3;
            for (auto& cnt : contours) {
                std::vector<cv::Point> kept;
                for (auto& pt : cnt) {
                    if (pt.x > margin && pt.x < size-margin &&
                        pt.y > margin && pt.y < size-margin)
                    {
                        kept.push_back(pt);
                    }
                }
                if (kept.size() >= 3)
                    cleaned.push_back(kept);
            }

            if (!cleaned.empty()) {
                // pick largest
                auto bestCnt = *std::max_element(
                    cleaned.begin(), cleaned.end(),
                    [](auto &a, auto &b){
                        return cv::contourArea(a) < cv::contourArea(b);
                    });
                // build hull into separate vector
                std::vector<cv::Point> hull;
                cv::convexHull(bestCnt, hull);
                if (hull.size() >= 5) {
                    cv::RotatedRect e = cv::fitEllipse(hull);
                    if (e.center.x > 0 && e.center.x < size &&
                        e.center.y > 0 && e.center.y < size)
                    {
                        ellipses.push_back(e);
                        validEllipse.push_back(true);
                        continue;
                    }
                }
            }

            // fallback: no valid ellipse
            ellipses.emplace_back();
            validEllipse.push_back(false);
            std::cout << "no eyes " << frameIdx << "\n";
        }

        // score & pick best ellipse
        std::vector<double> scores(3,0.0);
        for (int i = 0; i < 3; ++i) {
            double percent = 0.0, roundness = 0.0;
            if (validEllipse[i]) {
                auto &e = ellipses[i];
                cv::Mat eMask = cv::Mat::zeros(maskImages[i].size(), CV_8U);
                cv::ellipse(eMask, e, cv::Scalar(255), -1);
                double white = cv::countNonZero(maskImages[i] & eMask);
                double total = cv::countNonZero(eMask);
                percent   = (total > 0 ? (white/total)*100.0 : 0.0);
                roundness = (e.size.width / e.size.height);
            }
            scores[i] = percent + roundness*100.0;
        }
        int bestIdx = std::distance(scores.begin(),
                          std::max_element(scores.begin(), scores.end()));
        cv::RotatedRect bestEllipse = ellipses[bestIdx];
        bool hasBest = validEllipse[bestIdx];

        double prevArea = hasPrevEllipse
            ? (prevEllipse.size.width * prevEllipse.size.height) : 0.0;
        double bestArea = hasBest
            ? (bestEllipse.size.width * bestEllipse.size.height) : 0.0;

        if (!hasBest && hasPrevEllipse) {
            bestEllipse = prevEllipse;
            hasBest     = true;
        } else if (hasBest) {
            prevEllipse    = bestEllipse;
            hasPrevEllipse = true;
        }

        bool goodFrame = true;
        if (hasPrevEllipse && hasBest) {
            double dist = cv::norm(prevEllipse.center - bestEllipse.center);
            if (dist > bestEllipse.size.width) {
                bestEllipse = prevEllipse;
                goodFrame   = false;
                std::cout << "using prev " << frameIdx << "\n";
            }
            if (bestArea < prevArea*0.95 || bestArea > prevArea*1.10) {
                bestEllipse = prevEllipse;
                goodFrame   = false;
                std::cout << "using prev " << frameIdx << "\n";
            }
        }

        // draw
        if (hasBest) {
            cv::ellipse(eyeCrop, bestEllipse, cv::Scalar(200,190,140), 2);
            cv::circle(eyeCrop, bestEllipse.center, 2,
                       cv::Scalar(255,200,100), -1);
        }

        // paste back
        cv::Mat dstROI = frame(cv::Rect(x,y,w,h));
        eyeCrop(cv::Rect(0,0,w,h)).copyTo(dstROI);

        cv::imshow("Processed Frame", frame);

        if (goodFrame) {
            cv::imwrite("output/" + std::to_string(frameIdx) + ".png", toSave);
            csv << frameIdx << ","
                << x << "," << y << "," << w << "," << h << ","
                << bestEllipse.center.x << "," << bestEllipse.center.y << ","
                << bestEllipse.size.width << "," << bestEllipse.size.height << ","
                << bestEllipse.angle << "\n";
        }

        ++frameIdx;
        if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }

    auto tEnd = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<
        std::chrono::duration<double>>(tEnd - tStart).count();
    std::cout << "FPS: " << (frameIdx / elapsed) << "\n";

    cap.release();
    cv::destroyAllWindows();
    csv.close();
    return 0;
}