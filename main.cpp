#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>

using namespace std::chrono;

// Function to resize an image to a given size
cv::Mat resizeImage(const cv::Mat& image, int width, int height)
{
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));
    return resizedImage;
}

// Function to compute the mean value of an image
double computeMean(const cv::Mat& image)
{
    cv::Scalar meanVal = cv::mean(image);
    return meanVal[0];
}

// Function to compute the median value of an image
double computeMedian(const cv::Mat& image)
{
    cv::Mat sortedImage;
    cv::sort(image, sortedImage, cv::SORT_EVERY_COLUMN | cv::SORT_ASCENDING);
    int totalPixels = image.rows * image.cols;

    double medianValue;
    if (totalPixels % 2 == 0)
    {
        int index1 = (totalPixels / 2) - 1;
        int index2 = index1 + 1;
        medianValue = (sortedImage.at<uchar>(index1) + sortedImage.at<uchar>(index2)) / 2.0;
    }
    else
    {
        int index = (totalPixels / 2);
        medianValue = sortedImage.at<uchar>(index);
    }

    return medianValue;
}

// Function to compute the histogram of an image
cv::Mat computeHistogram(const cv::Mat& image)
{
    int numBins = 256; // Number of bins for the histogram
    int histSize[] = { numBins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    int channels[] = { 0 }; // Compute histogram only for the first channel (grayscale image)

    cv::Mat histogram;
    cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 1, histSize, ranges);

    return histogram;
}

// Function to compute the standard deviation of an image
double computeStandardDeviation(const cv::Mat& image)
{
    double mean = computeMean(image);
    cv::Scalar meanSquaredDiff = cv::mean((image - mean).mul(image - mean));
    return std::sqrt(meanSquaredDiff[0]);
}

// Function to compute the Hu Moments of an image
cv::Mat computeHuMoments(const cv::Mat& image)
{
    cv::Moments moments = cv::moments(image);
    cv::Mat huMoments;
    cv::HuMoments(moments, huMoments);
    return huMoments;
}

// Function to compute the Euclidean distance between two feature vectors
double computeDistance(const cv::Mat& queryFeatureVector, const cv::Mat& datasetFeatureVector)
{
    cv::Mat diff;
    cv::absdiff(queryFeatureVector, datasetFeatureVector, diff);
    cv::Mat squaredDiff = diff.mul(diff);
    cv::Scalar sum = cv::sum(squaredDiff);
    return std::sqrt(sum[0]);
}

std::vector<std::string> getFilesInDirectory(const std::string& dirPath)
{
    std::vector<std::string> fileNames;
    DIR* directory;
    struct dirent* entry;

    directory = opendir(dirPath.c_str());
    if (directory != nullptr)
    {
        while ((entry = readdir(directory)) != nullptr)
        {
            std::string fileName = entry->d_name;
            if (fileName != "." && fileName != "..")
            {
                fileNames.push_back(dirPath + fileName);
            }
        }
        closedir(directory);
    }

    return fileNames;
}

int main()
{
    auto start = high_resolution_clock::now();
    std::string datasetPath = "./dataset/images/"; // Path to the dataset images
    std::string queryImagePath = "000000124442.jpg"; // Path to the query image

    std::vector<std::string> imagePaths = getFilesInDirectory(datasetPath);

    // Load and resize the query image
    cv::Mat queryImage = cv::imread(queryImagePath, cv::IMREAD_GRAYSCALE);
    queryImage = resizeImage(queryImage, 500, 500);

    // Apply noise reduction techniques to the query image
    cv::Mat queryImageSmallBlur;
    cv::Mat queryImageLargeBlur;
    cv::Mat queryImageGaussianBlur;

    cv::blur(queryImage, queryImageSmallBlur, cv::Size(3, 3));
    cv::blur(queryImage, queryImageLargeBlur, cv::Size(9, 9));
    cv::GaussianBlur(queryImage, queryImageGaussianBlur, cv::Size(9, 9), 0);

    // Compute features for the query image
    double queryMean = computeMean(queryImage);
    double queryMedian = computeMedian(queryImage);
    double queryStdDev = computeStandardDeviation(queryImage);
    cv::Mat queryHuMoments = computeHuMoments(queryImage);
    cv::Mat queryHistogram = computeHistogram(queryImage);

    // Compute features for all dataset images
    std::vector<std::string> bestImagePaths;
    std::vector<double> bestImageDistances(20, std::numeric_limits<double>::max());

    #pragma omp parallel for
    for (int i = 0; i < imagePaths.size(); ++i)
    {
        // Load and resize the dataset image
        cv::Mat datasetImage = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
        datasetImage = resizeImage(datasetImage, 500, 500);

        // Apply noise reduction techniques to the dataset image
        cv::Mat datasetImageSmallBlur;
        cv::Mat datasetImageLargeBlur;
        cv::Mat datasetImageGaussianBlur;

        cv::blur(datasetImage, datasetImageSmallBlur, cv::Size(3, 3));
        cv::blur(datasetImage, datasetImageLargeBlur, cv::Size(9, 9));
        cv::GaussianBlur(datasetImage, datasetImageGaussianBlur, cv::Size(9, 9), 0);

        // Compute features for the dataset image
        double datasetMean = computeMean(datasetImage);
        double datasetMedian = computeMedian(datasetImage);
        double datasetStdDev = computeStandardDeviation(datasetImage);
        cv::Mat datasetHuMoments = computeHuMoments(datasetImage);
        cv::Mat datasetHistogram = computeHistogram(datasetImage);

        // Compute similarity using Euclidean distance
        double distance = computeDistance(cv::Mat(1, 1, CV_64F, &datasetMean), cv::Mat(1, 1, CV_64F, &queryMean))
            + computeDistance(cv::Mat(1, 1, CV_64F, &datasetMedian), cv::Mat(1, 1, CV_64F, &queryMedian))
            + computeDistance(cv::Mat(1, 1, CV_64F, &datasetStdDev), cv::Mat(1, 1, CV_64F, &queryStdDev))
            + computeDistance(datasetHuMoments, queryHuMoments)
            + computeDistance(datasetHistogram, queryHistogram);

        #pragma omp critical
        {
            // Update the best image paths if the current distance is smaller than the current best distances
            for (int j = 0; j < bestImageDistances.size(); ++j)
            {
                if (distance < bestImageDistances[j])
                {
                    bestImagePaths.insert(bestImagePaths.begin() + j, imagePaths[i]);
                    bestImagePaths.resize(20);
                    bestImageDistances.insert(bestImageDistances.begin() + j, distance);
                    bestImageDistances.resize(20);
                    break;
                }
            }
        }
        
    }

    // Save the names of the best 20 images to a file
    std::ofstream outputFile("best_images.txt");
    if (outputFile.is_open())
    {
        for (const auto& imagePath : bestImagePaths)
        {
            outputFile << imagePath << std::endl;
        }
        outputFile.close();
    }
    else
    {
        std::cerr << "Failed to open the output file." << std::endl;
        return 1;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken in Serial mode : " << duration.count() << " microseconds\n";

    return 0;
}