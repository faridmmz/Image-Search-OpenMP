# Parallel Image Search using OpenMP

![Photo by <a href="https://unsplash.com/@brostvarta?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Francesco Vantini</a> on <a href="https://unsplash.com/photos/ZavLsrP4CDI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
   ](https://github.com/faridmmz/Image-Search-OPenMP/blob/main/README_image.jpg "Photo by Francesco Vantini on Unsplash")

## Contributors

- Faridreza Momtazzandi
- Mahya Ehsanimehr
- Ali Mojahed

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-Overview)
- [Code Structure](#code-Structure)
- [Execution and Results](#execution-and-Results)
- [Documentation](#documentation)
- [Getting Started](#getting-Started)
- [Conclusion](#conclusion)

## Introduction
This repository contains a parallel programming project aimed at finding similar images using an image query from a dataset of random and unrelated images. The project involves image processing, feature extraction, and utilizing parallelization techniques for efficient computation using multi-core processing.

## Project Overview
The primary goal of the project is to find similar images to a given query image within a dataset. This involves several important steps:
- Applying noise reduction techniques to images
- Calculating various image features such as mean, median, standard deviation, Hu moments, and histogram
- Utilizing the calculated features to construct feature vectors for images
- Computing the Euclidean distance between feature vectors to measure image similarity
- Implementing parallelization to expedite the computation process using OpenMP

## Code Structure
The core functionality of the project is implemented in `main.cpp`, which is the entry point of the program. The code is organized into functions for image processing, feature calculation, distance computation, and parallelization using OpenMP. Additionally, a `Makefile` is provided to compile the code.

## Execution and Results
The project was executed on Ubuntu 20.04 with an Intel Core i5-6200U CPU @ 2.30GHz, 2.4 GHz, and 8 GB of RAM. The program's performance was evaluated under different scenarios and configurations. Execution times were recorded for varying numbers of threads, and the results were visualized using charts. The hardware specifications used for testing are mentioned in the document.

## Documentation
A detailed documentation file named `multicoreproject-omp-doc.pdf` is provided in the repository. This document outlines the project's goals, code review, presentation of results, and hardware specifications used for testing. It provides valuable insights into the project's design, execution, and outcomes.

## Getting Started
To compile the code, use the provided Makefile:
```bash
make
```
To run the compiled program:
```
./main
```
Make sure to have the required OpenCV library installed.

## Conclusion
The Multi-Core Programming project demonstrates the application of parallel programming concepts to image processing and similarity computation. The team successfully implemented parallelization using OpenMP to achieve faster execution times when finding similar images within a dataset. The documentation and code structure provide a comprehensive overview of the project's objectives, methodologies, and outcomes.
